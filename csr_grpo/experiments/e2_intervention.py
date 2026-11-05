"""E2 — reward-source intervention (H2/C2).

Two protocols, deliberately separated (paper §E2):

**Protocol A — single-step local mechanism.**
    Freeze actor parameters, the rollout batch and the Adam state.  For each
    reward source, restore the frozen state, score the *same* rollouts, take one
    real optimiser step, and read ``Δθ = θ_after − θ_before``.  Report

    * ``CosSim(Δθ^(i), Δθ^gold)``
    * ``ΔL_old^(1st) = ∇_θ L_old(θ)ᵀ Δθ^(i)``  (>0 ⇒ this step raises old-task loss)
    * ``‖P_{U_old} Δθ^(i)‖ / ‖Δθ‖`` as a sign-free auxiliary

    Because Δθ comes from the real optimizer, Adam momentum and clipping are
    inside the measurement, which is exactly why the plan replaces the
    basis-dependent ``⟨g, U_old⟩`` with this scalar.

**Protocol B — multi-step branch resampling.**
    Each arm runs K steps, re-sampling rollouts from its *own* updated policy
    every step.  Measures the closed-loop cumulative effect: paired
    ``ΔAcc_old`` and ``ΔAcc_new`` against the branch's own start.

The gold arm is a declared diagnostic arm; its L3 reads are logged separately
from training-time gold and never enter a method's training protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import torch

from ..compat import to_np, to_tensor
from ..config.schema import Config
from ..diagnostics.fidelity import per_prompt_stats
from ..methods.base import make_method
from ..rewards.advantage import compute_advantages
from ..rewards.sources import RewardSource
from ..supervision.ledger import Role, SupervisionLedger
from ..training.runner import Runner, StageContext
from ..training.state import RolloutBatch, TrainingState


# ---------------------------------------------------------------------------
# old-task loss and its gradient
# ---------------------------------------------------------------------------


def old_task_loss(runner: Runner, prompt_ids: torch.Tensor,
                  answers: torch.Tensor, *, audit_stage: int) -> torch.Tensor:
    """``L_old(θ)``: gold cross-entropy of *sampled* old-task answers.

    The target must be the gold correctness of the answers the actor actually
    produced, not the gold token itself — the latter is a one-hot target on the
    argmax, whose gradient is structurally near zero and would make
    ``ΔL_old^(1st)`` meaningless.  Gold labels are an audit read.
    """
    correct = runner.score_with_gold(audit_stage, prompt_ids, answers).detach()
    # ``answers`` may be (P, G); the actor scores one prompt per row, so flatten
    # the group axis and repeat each prompt for its group.
    if answers.dim() > 1:
        p, g = answers.shape
        prompt_ids = prompt_ids.repeat_interleave(g, dim=0)
        answers = answers.reshape(-1)
        correct = correct.reshape(-1)
    logits = runner.model.actor_logits(prompt_ids)
    logp = torch.log_softmax(logits, dim=-1)
    tok = answers.long()
    nll = -logp.gather(1, tok[:, None]).squeeze(1)
    # L_old is a *loss*: higher when the model is *worse* on old-task answers.
    # weight +1 on correct answers (NLL rises when the model forgets them) and
    # -1 on wrong answers (NLL falls when the model forgets them, but that is
    # not damage).  This makes ΔL_old^(1st) = ∇L_old^T Δθ > 0 iff the step
    # degrades old-task performance — the sign the spec requires (§H2 / C2).
    # The previous sign (1 - 2*correct) was inverted: it was a reward signal
    # (higher when the model was *better*), so dL_old_1st > 0 wrongly meant
    # the step *improved* old-task performance.
    return (nll * (2.0 * correct - 1.0)).mean()


def old_task_gradient(runner: Runner, prompt_ids: torch.Tensor,
                      answers: torch.Tensor, *, audit_stage: int
                      ) -> dict[str, torch.Tensor]:
    """``∇_θ L_old(θ)`` keyed exactly like ``TrainingState.param_delta``.

    The key convention matters: ``param_delta`` walks ``named_parameters()``
    (``actor_bank.actor.B``), so gradients must be collected the same way —
    using ``view_parameters`` names (``actor.B``) silently makes every inner
    product empty and returns NaN.
    """
    runner.model.set_stage(audit_stage)
    runner.optimizer.zero_grad(set_to_none=True)
    loss = old_task_loss(runner, prompt_ids, answers, audit_stage=audit_stage)
    loss.backward()
    trainable = {id(p) for p in runner.actor_params}
    grads = {
        name: p.grad.detach().clone()
        for name, p in runner.model.named_parameters()
        if p.grad is not None and id(p) in trainable
    }
    runner.optimizer.zero_grad(set_to_none=True)
    return grads


def cosine_similarity(a: dict[str, torch.Tensor],
                      b: dict[str, torch.Tensor]) -> float:
    """Flattened cosine over the union of parameter tensors."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan")
    va = torch.cat([a[k].reshape(-1).float() for k in keys])
    vb = torch.cat([b[k].reshape(-1).float() for k in keys])
    denom = va.norm() * vb.norm()
    return float((va @ vb) / denom) if denom > 0 else float("nan")


def first_order_delta(grads: dict[str, torch.Tensor],
                      delta: dict[str, torch.Tensor]) -> float:
    """``∇_θ L_old^ᵀ Δθ`` — the sign-stable first-order damage scalar."""
    keys = sorted(set(grads) & set(delta))
    if not keys:
        return float("nan")
    return float(sum((grads[k].reshape(-1).float() * delta[k].reshape(-1).float()).sum()
                     for k in keys))


def _flatten_vec(vecs: dict[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
    """Concatenate per-parameter tensors in a fixed key order."""
    return torch.cat([vecs[k].reshape(-1).float() for k in keys])


def _mean_gradients(grad_matrix: torch.Tensor, keys: Sequence[str],
                    runner: Runner) -> dict[str, torch.Tensor]:
    """Rebuild a per-parameter gradient dict from the mean of stacked rows.

    The matrix stores one row per prompt; ``ΔL_old^(1st)`` is a scalar product,
    so the natural companion gradient is the mean old-task gradient vector.
    """
    mean_row = grad_matrix.mean(dim=0)
    shapes = {n: p.shape for n, p in runner.model.named_parameters()}
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name in keys:
        numel = int(torch.tensor(shapes[name]).prod().item())
        out[name] = mean_row[offset: offset + numel].reshape(shapes[name]).clone()
        offset += numel
    if offset != mean_row.numel():  # pragma: no cover - structural guard
        raise RuntimeError(
            f"gradient layout mismatch: consumed {offset} of {mean_row.numel()}"
        )
    return out


def per_prompt_old_gradients(runner: Runner, prompt_ids: torch.Tensor,
                             answers: torch.Tensor, *, audit_stage: int,
                             max_rows: int = 16
                             ) -> tuple[torch.Tensor, list[str]]:
    """``(N, D)`` matrix of old-task gradients, one row per prompt.

    A single gradient gives a rank-1 subspace, so projecting onto it would
    trivially absorb any Δθ and the ratio would be uninformative.  Stacking
    per-prompt gradients (capped at ``max_rows`` for cost) yields a genuine
    estimate of the old-task gradient subspace ``U_old``.
    """
    keys = sorted(n for n, _ in runner.model.named_parameters())
    trainable = {id(p) for p in runner.actor_params}
    keys = [k for k in keys if id(dict(runner.model.named_parameters())[k]) in trainable]

    runner.model.set_stage(audit_stage)
    rows: list[torch.Tensor] = []
    n = min(len(prompt_ids), max_rows)
    for i in range(n):
        runner.optimizer.zero_grad(set_to_none=True)
        # score this single prompt's whole group
        pid = prompt_ids[i: i + 1]
        if answers.dim() > 1:
            ans = answers[i: i + 1]
        else:
            ans = answers[i: i + 1]
        loss = old_task_loss(runner, pid, ans, audit_stage=audit_stage)
        loss.backward()
        grads = {name: p.grad.detach().clone()
                 for name, p in runner.model.named_parameters()
                 if p.grad is not None and id(p) in trainable}
        rows.append(_flatten_vec(grads, keys))
    runner.optimizer.zero_grad(set_to_none=True)
    return torch.stack(rows), keys


def projection_energy_ratio(delta: dict[str, torch.Tensor],
                            grad_matrix: torch.Tensor,
                            keys: Sequence[str],
                            rank: int = 8) -> float:
    """``‖P_{U_old} Δθ‖ / ‖Δθ‖`` with ``U_old`` from an SVD of per-prompt grads."""
    have = [k for k in keys if k in delta]
    if not have:
        return float("nan")
    if len(have) != len(keys):
        # key order must match the gradient matrix exactly
        return float("nan")
    d = _flatten_vec(delta, keys).reshape(1, -1)
    if grad_matrix.shape[1] != d.shape[1]:
        return float("nan")
    if grad_matrix.shape[0] < 2:
        return float("nan")
    _, sv, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
    # keep only directions with non-trivial energy
    k = int(min(rank, int((sv > 1e-12 * sv.max()).sum().item())))
    if k == 0:
        return float("nan")
    u = vh[:k].t()
    proj = d @ u @ u.t()
    denom = d.norm()
    return float(proj.norm() / denom) if denom > 0 else float("nan")



def _reset_optimizer_moments(runner: Runner) -> None:
    """Zero Adam's moments before a protocol-A step.

    "Lock the Adam state" means every arm must step from the *same* optimizer
    state; the snapshot does that, but the snapshot's moments are whatever the
    preceding training left behind.  A step with a zero gradient then still
    moves the parameters (measured: ``grad_norm = 0`` but ``‖Δθ‖ = 8.1e-5``),
    so ``Δθ`` would contain an update no reward source produced.  The gold arm
    in a real run had ``adv_std = 0.0`` and a non-zero ``Δθ`` for exactly this
    reason.

    Zeroing the moments makes ``Δθ`` a function of *this* step's gradient
    only, which is what ``CosSim(Δθ, Δθ^gold)`` and
    ``ΔL_old^(1st) = ∇L_old^T Δθ`` are claims about.
    """
    for group in runner.optimizer.param_groups:
        for p in group["params"]:
            st = runner.optimizer.state.get(p)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key].zero_()
            if "step" in st:
                if isinstance(st["step"], torch.Tensor):
                    st["step"].zero_()
                else:
                    st["step"] = 0


def _arm_advantage_config(runner: Runner, cfg: Config):
    """Advantage config for the current E2 arm.

    ``degenerate_zero`` is a property of the *reward source* here, not of the
    method: arm (iv) is defined as "current judge, advantage forced to 0 on
    groups the judge believes are all-good/all-bad".  Deriving it from
    ``plan.degenerate_zero`` (False for dual_adapter) made arm (iv) an exact
    duplicate of arm (i) -- identical Δθ, identical CosSim, identical
    ΔL_old^(1st) -- while still reporting four arms.
    """
    src = runner.plan.reward_source
    base = runner.plan.advantage_config(eps=cfg.train.adv_eps)
    if src is RewardSource.DEGENERATE_ZEROED:
        return replace(base, degenerate_zero=True, degenerate_source="judge")
    return base


# ---------------------------------------------------------------------------
# Protocol A
# ---------------------------------------------------------------------------


@dataclass
class ProtocolAResult:
    arms: dict[str, dict] = field(default_factory=dict)
    gold_arm: str = "gold"
    rewards: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"arms": self.arms, "gold_arm": self.gold_arm}


def run_protocol_a(cfg: Config, ledger: SupervisionLedger, *,
                   stage: int = 2, audit_stage: int = 1,
                   step: int = 50, seed: int = 0,
                   arms: Sequence[str] | None = None,
                   n_prompts: int = 8, n_old_prompts: int = 16
                   ) -> ProtocolAResult:
    """Single-step intervention with a frozen actor / rollout / Adam state."""
    if arms is None:
        arms = tuple(cfg.exp.e2_arms)
    import copy

    base_cfg = copy.deepcopy(cfg)
    base_cfg.method = "dual_adapter"
    base_cfg.reward_source = RewardSource.SELF.value
    base_cfg.seed = seed
    method = make_method(base_cfg)
    runner = Runner(base_cfg, method.plan(), ledger)
    runner.warmup_judge(audit_stage, n_rounds=40)
    runner.freeze_judge()

    ctx = runner.make_context(stage, n_prompts=64, seed=10 + stage,
                              audit_stage=audit_stage, audit_seed=base_cfg.exp.audit_seed)
    # drive the actor a little so Δθ is not a first-step special case
    for s in range(step):
        runner.train_step(ctx, s)

    # --- freeze everything the protocol requires --------------------------
    frozen: TrainingState = runner.snapshot(tag="protocolA")
    G = base_cfg.train.group_size
    pid = ctx.prompts[:n_prompts]
    runner.model.set_stage(stage)
    with torch.no_grad():
        seqs, mask, old_lp = runner.model.sample(pid, G, generator=runner.rng)
    answers = seqs[:, -1].reshape(n_prompts, G)
    batch = RolloutBatch(
        task_ids=[ctx.task_ids[i] for i in range(n_prompts)],
        sequences=seqs.view(n_prompts, G, -1),
        attention_mask=mask.view(n_prompts, G, -1),
        old_logprob=old_lp.view(n_prompts * G, -1),
        prompt_len=pid.shape[1], sampled_at_step=step,
    ).collapse()
    rollout_fp = batch.fingerprint()

    # old-task gradient at θ (audit read; outside any training scope).
    # The loss must be evaluated on *sampled* old-task answers, so draw them
    # from the actor with the old stage active.
    # cap to the actual audit set: a default n_old_prompts larger than
    # cfg.data.audit_prompts would silently slice fewer rows and the reshape
    # below would raise shape mismatch
    n_old_prompts = min(n_old_prompts, len(ctx.audit_prompts))
    old_pid = ctx.audit_prompts[:n_old_prompts]
    runner.model.set_stage(audit_stage)
    with torch.no_grad():
        old_seqs, _, _ = runner.model.sample(old_pid, G, generator=runner.rng)
    old_ans = old_seqs[:, -1].reshape(n_old_prompts, G)
    old_grad_matrix, old_keys = per_prompt_old_gradients(
        runner, old_pid, old_ans, audit_stage=audit_stage)
    # the first-order term uses the mean old-task gradient per parameter
    old_grads = _mean_gradients(old_grad_matrix, old_keys, runner)

    deltas: dict[str, dict[str, torch.Tensor]] = {}
    diags: dict[str, dict] = {}

    for arm in arms:
        runner.restore(frozen)
        _reset_optimizer_moments(runner)
        src = RewardSource(arm)
        runner.plan.reward_source = src
        runner.adv_cfg = _arm_advantage_config(runner, base_cfg)

        gold_control = src is RewardSource.GOLD
        with ledger.training_scope(gold_control=gold_control):
            rewards = runner.reward_matrix(stage, pid, answers,
                                           purpose=f"protocolA:{arm}")
            adv = compute_advantages(to_np(rewards), cfg=runner.adv_cfg)
            res = runner.trainer.step_from_rollout(
                batch, to_tensor(adv.reshape(-1)).float(), record_delta=True)
        delta = res.delta_signature
        deltas[arm] = delta
        diags[arm] = {
            "cos_vs_gold": cosine_similarity(delta, deltas.get("gold", {})),
            "dL_old_1st": first_order_delta(old_grads, delta),
            "proj_ratio": projection_energy_ratio(delta, old_grad_matrix, old_keys),
            "delta_norm": res.delta_norm,
            "grad_norm": res.grad_norm,
            "adv_std": res.adv_std,
            "rollout_fingerprint": rollout_fp,
        }

    runner.restore(frozen)
    gold_delta = deltas.get("gold", {})
    for arm, d in diags.items():
        d["cos_vs_gold"] = (1.0 if arm == "gold" else
                            cosine_similarity(deltas[arm], gold_delta))

    # The gold arm is the *reference direction*.  If its own step produced no
    # update, every CosSim is undefined and the whole protocol-A readout is
    # vacuous -- that happens when the gold rewards are constant within every
    # group (all-correct or all-wrong), i.e. exactly the degenerate case the
    # paper is about.  Say so instead of reporting NaN.
    gold_norm = float(np.sqrt(sum(float((v ** 2).sum())
                                  for v in gold_delta.values()))) if gold_delta else 0.0
    status = "gold_arm_inactive" if (not np.isfinite(gold_norm) or gold_norm <= 1e-12) else "ok"
    for d in diags.values():
        d["gold_delta_norm"] = gold_norm
        d["status"] = status
        if status != "ok" and d is not diags.get("gold"):
            d["cos_vs_gold"] = float("nan")
    return ProtocolAResult(arms=diags)


# ---------------------------------------------------------------------------
# Protocol B
# ---------------------------------------------------------------------------


@dataclass
class BranchResult:
    arm: str
    acc_old: list[float]
    acc_new: list[float]
    acc_old_start: float
    acc_new_start: float

    @property
    def d_acc_old(self) -> float:
        return self.acc_old[-1] - self.acc_old_start

    @property
    def d_acc_new(self) -> float:
        return self.acc_new[-1] - self.acc_new_start

    def as_dict(self) -> dict:
        return {"arm": self.arm, "d_acc_old": self.d_acc_old,
                "d_acc_new": self.d_acc_new,
                "acc_old": self.acc_old, "acc_new": self.acc_new}


def run_protocol_b(cfg: Config, ledger: SupervisionLedger, *,
                   stage: int = 2, audit_stage: int = 1, k_steps: int = 20,
                   seed: int = 0,
                   arms: Sequence[str] = ("self", "protected", "gold",
                                          "degenerate_zeroed"),
                   n_eval: int = 24) -> dict[str, BranchResult]:
    """K-step branches, each re-sampling from its own updated policy."""
    import copy

    base_cfg = copy.deepcopy(cfg)
    base_cfg.method = "dual_adapter"
    base_cfg.reward_source = RewardSource.SELF.value
    base_cfg.seed = seed
    method = make_method(base_cfg)
    runner = Runner(base_cfg, method.plan(), ledger)
    runner.warmup_judge(audit_stage, n_rounds=40)
    runner.freeze_judge()

    ctx = runner.make_context(stage, n_prompts=64, seed=10 + stage,
                              audit_stage=audit_stage, audit_seed=base_cfg.exp.audit_seed)
    frozen = runner.snapshot(tag="protocolB")

    # fixed evaluation sets, identical across branches
    ev_new_ids = ctx.prompts[:n_eval]
    ev_old_ids = ctx.audit_prompts[:n_eval]
    G = base_cfg.train.group_size

    def accuracy(ids: torch.Tensor, eval_stage: int) -> float:
        """Pass@1: the actor's single greedy answer per prompt.

        Sampling ``G`` answers and taking ``max`` over the group measures
        pass@G, which saturates near 1.0 and washes out the very differences
        protocol B is meant to detect.  Greedy decoding is the honest pass@1
        analogue and is deterministic, so paired comparisons across branches
        are not confounded by sampling noise.
        """
        runner.model.set_stage(eval_stage)
        with torch.no_grad():
            logits = runner.model.actor_logits(ids)
            pred = logits.argmax(dim=-1)
        gold = to_np(runner.score_with_gold(eval_stage, ids, pred))
        return float(gold.mean())

    start_new = accuracy(ev_new_ids, stage)
    start_old = accuracy(ev_old_ids, audit_stage)

    out: dict[str, BranchResult] = {}
    for arm in arms:
        runner.restore(frozen)
        src = RewardSource(arm)
        runner.plan.reward_source = src
        runner.adv_cfg = _arm_advantage_config(runner, base_cfg)
        acc_new, acc_old = [], []
        gold_control = src is RewardSource.GOLD
        for s in range(k_steps):
            runner.train_step(ctx, s)           # re-samples on-policy each step
            if (s + 1) % max(1, k_steps // 5) == 0 or s == k_steps - 1:
                acc_new.append(accuracy(ev_new_ids, stage))
                acc_old.append(accuracy(ev_old_ids, audit_stage))
        out[arm] = BranchResult(arm=arm, acc_old=acc_old, acc_new=acc_new,
                                acc_old_start=start_old, acc_new_start=start_new)
    runner.restore(frozen)
    return out


def paired_branch_differences(branches: dict[str, BranchResult],
                              reference: str = "gold") -> dict[str, dict]:
    """``ΔAcc`` contrasts between each arm and the reference branch."""
    ref = branches[reference]
    out = {}
    for arm, b in branches.items():
        if arm == reference:
            continue
        out[arm] = {
            "d_acc_old_vs_gold": b.d_acc_old - ref.d_acc_old,
            "d_acc_new_vs_gold": b.d_acc_new - ref.d_acc_new,
        }
    return out


@dataclass
class E2Result:
    protocol_a: dict
    protocol_b: dict | None = None
    contrasts: dict | None = None

    def as_dict(self) -> dict:
        return {"protocol_a": self.protocol_a,
                "protocol_b": self.protocol_b,
                "contrasts": self.contrasts}


def run_e2(cfg: Config, ledger: SupervisionLedger, *, stage: int = 2,
           audit_stage: int = 1, protocol_b: bool = True,
           k_steps: int = 20, seed: int = 0,
           checkpoints: tuple[int, ...] | None = None) -> E2Result:
    """Run protocol A at each checkpoint and protocol B once.

    ``checkpoints`` overrides ``cfg.exp.e2_checkpoints``.  Each element is the
    number of training steps after which the frozen single-step measurement is
    taken (protocol A).  Protocol B's K-step resampling runs at the *last*
    checkpoint only (it already accumulates K full steps internally).
    """
    steps = checkpoints if checkpoints is not None else tuple(cfg.exp.e2_checkpoints)
    # run protocol A at every checkpoint
    pa_by_step: dict[int, dict] = {}
    for s in steps:
        a = run_protocol_a(cfg, ledger, stage=stage, audit_stage=audit_stage,
                           seed=seed, step=s)
        pa_by_step[s] = a.as_dict()
    if not protocol_b:
        return E2Result(protocol_a=pa_by_step)
    b = run_protocol_b(cfg, ledger, stage=stage, audit_stage=audit_stage,
                       k_steps=k_steps, seed=seed)
    return E2Result(protocol_a=pa_by_step,
                    protocol_b={k: v.as_dict() for k, v in b.items()},
                    contrasts=paired_branch_differences(b))


def run_e2_prime(cfg: Config, ledger: SupervisionLedger, *,
                 protocol_b: bool = True, k_steps: int = 20,
                 seed: int = 0,
                 checkpoints: tuple[int, ...] | None = None) -> E2Result:
    """Independent held-out E2 confirmation on the Stage 2 → 3 transition.

    Mirrors ``run_e1_prime``: the same protocol runs on a transition the model
    has not seen, so the held-out result is genuinely independent of any
    hyper-parameter search done on Stage 1 → 2.  The audit_stage is 2 (the
    frozen old-task probe) and the target stage is 3 (the new task).
    """
    return run_e2(cfg, ledger, stage=3, audit_stage=2,
                  protocol_b=protocol_b, k_steps=k_steps, seed=seed,
                  checkpoints=checkpoints)
