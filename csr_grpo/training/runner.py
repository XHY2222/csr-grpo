"""A stage-runner that wires the components together without hiding anything.

Ownership boundaries, deliberately strict:

* **Training steps** live in ``GRPOTrainer.step_from_rollout`` and open a normal
  training scope; any L3 read from a normal method is a hard error.
* **Gold-control arms** (E1's Gold DID control, E2's diagnostic gold reward,
  Row 10) call the same step function with ``gold_control=True``.  The ledger
  still records every L3 read and reports it separately, so the paper can state
  ``gold_calls_in_training = 0`` for all trained methods.
* **Measurements** need gold labels to *evaluate* what happened.  They run after
  the step, under an ``AUDIT_REPORTING`` grant, outside any training scope.
  The ledger therefore alternates: train (no gold) → measure (audited gold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..backends import build_backend
from ..backends.protocol import check_model, check_world
from ..backends.toy import ToyModel, ToyWorld
from ..compat import to_np, to_tensor
from ..config.schema import Config
from ..diagnostics.dual_line import LINE_AUDIT, LINE_ONPOLICY, DualLine
from ..diagnostics.fidelity import PerPromptStats, per_prompt_stats
from ..methods.base import MethodPlan
from ..rewards.advantage import compute_advantages
from ..rewards.sources import RewardSource
from ..supervision.ledger import Level, Role, SupervisionLedger
from ..training.state import ModuleState, RolloutBatch, TrainingState
from ..training.trainer import GRPOTrainer


@dataclass
class StageContext:
    stage: int
    prompts: torch.Tensor                     # current-stage prompts (N, L)
    task_ids: list[str]
    audit_prompts: torch.Tensor | None = None  # old-stage P_audit prompts (P, L)
    audit_task_ids: list[str] | None = None


@dataclass
class Observation:
    step: int
    line: str
    values: dict[str, float]
    n_informative: int = 0
    n_degenerate: int = 0


@dataclass
class StageRun:
    stage: int
    observations: list[Observation] = field(default_factory=list)
    snapshots: dict[str, TrainingState] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def series(self, line: str, metric: str) -> tuple[list[int], list[float]]:
        steps, vals = [], []
        for o in self.observations:
            if o.line == line:
                steps.append(o.step)
                vals.append(o.values.get(metric, float("nan")))
        return steps, vals


class Runner:
    """Owns the model, judge, trainer, ledger and the measurement lines."""

    def __init__(self, cfg: Config, plan: MethodPlan, ledger: SupervisionLedger,
                 *, world: ToyWorld | None = None, model: object | None = None):
        self.cfg = cfg
        self.plan = plan
        self.ledger = ledger
        self.world = world or ToyWorld()
        # ``model`` is injected so a real backend can run this loop unchanged.
        # Both halves are checked against the protocol *before* any work is
        # done: a missing method used to surface as an AttributeError on the
        # first training step, after warm-up and (on the cluster) model loading.
        check_world(self.world)
        if model is not None:
            self.model = model
        elif cfg.model.backend == "toy" or cfg.model.backend is None:
            self.model = ToyModel(self.world, plan.spec, seed=cfg.seed)
        else:
            # A non-toy backend must be fully constructed (weights loaded) by
            # the caller and injected via ``model=``.  Constructing it here
            # would require a model path, a tokenizer path, device assignments,
            # and a gold verifier -- all caller-owned decisions.  The backend
            # name in the config serves as a documentation signal that this run
            # is meant for the cluster; a missing injection is caught below.
            raise ValueError(
                f"cfg.model.backend={cfg.model.backend!r} requires an "
                "explicit ``model=`` argument: load the backend, call "
                "``check_model(backend)`` yourself, and pass it to Runner."
            )
        check_model(self.model)
        self.model.set_stage(1)

        self.actor_params = [p for _, p in
                             self.model.view_parameters("actor")]
        if not self.actor_params:
            raise RuntimeError("method has no trainable actor adapters")
        actor_lr = cfg.train.sft_lr if plan.supervised_only else cfg.train.lr
        self.optimizer = torch.optim.Adam(
            self.actor_params, lr=actor_lr,
            betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
            eps=cfg.train.adam_eps,
        )

        # Warm-up always needs judge-side capacity, even when the method
        # freezes the judge inside a stage (Dual-Adapter).  "Frozen in stage"
        # is enforced by never stepping this optimizer between stage start and
        # stage end, not by removing the parameters.
        self.judge_params = [p for _, p in self.model.view_parameters("judge")]
        self.judge_optimizer = (
            torch.optim.Adam(self.judge_params, lr=cfg.train.warmup_lr)
            if self.judge_params else None
        )
        self.judge_trainable_in_stage = plan.judge_trainable_in_stage

        self.trainer = GRPOTrainer(
            self.model, self.optimizer, ledger,
            grad_clip=cfg.train.grad_clip, kl_coef=cfg.train.kl_coef,
            device=cfg.train.device,
            ref_kl_clip=cfg.train.ref_kl_clip,
        )
        self.adv_cfg = plan.advantage_config(eps=cfg.train.adv_eps)
        self.dual_line = DualLine(tau=cfg.train.degenerate_tau)
        self.rng = torch.Generator().manual_seed(cfg.seed)
        self.protected_judge: ModuleState | None = None
        self.prev_stage_judge: ModuleState | None = None   # θ_J^(t-1,end) for E_w / H6
        # Row 4: number of demonstrations prepended to the frozen judge prompt
        self.icl_shots: int = int(getattr(plan, "icl_shots", 0) or 6)
        self._history: list[StageRun] = []

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @property
    def is_gold_arm(self) -> bool:
        return self.plan.reward_source is RewardSource.GOLD

    def stage_prompts(self, stage: int, n: int, seed: int
                      ) -> tuple[torch.Tensor, list[str]]:
        return self.world.make_prompts(stage, n, seed=seed)

    def make_context(self, stage: int, *, n_prompts: int, seed: int,
                     audit_stage: int | None = None,
                     audit_seed: int | None = None) -> StageContext:
        """Build a stage context, optionally with its ``P_audit`` prompt set.

        ``audit_seed`` must be supplied by any caller that compares two arms
        paired on ``P_audit``.  The default derives it from ``seed``, which in
        turn is derived from the *target* stage — so two arms switching at
        different stages would otherwise measure different prompts, and a
        paired bootstrap would pair prompt *i* of one arm with a different
        question at index *i* of the other.
        """
        ids, tids = self.stage_prompts(stage, n_prompts, seed)
        ctx = StageContext(stage=stage, prompts=ids, task_ids=tids)
        if audit_stage is not None:
            a_ids, a_tids = self.stage_prompts(
                audit_stage, self.cfg.data.audit_prompts,
                audit_seed if audit_seed is not None else seed + 500)
            ctx.audit_prompts, ctx.audit_task_ids = a_ids, a_tids
        return ctx

    # ------------------------------------------------------------------
    # judge warm-up (L1/L2 only)
    # ------------------------------------------------------------------
    def warmup_judge(self, stage: int, n_rounds: int = 20,
                     n_per_round: int = 16) -> list[float]:
        """Route-A style weak-label training of the judge.

        Labels are synthetic: the "good" answer is the world's modal answer for
        the stage, the "bad" one is that answer + 1 (a rule perturbation).  This
        is L1 supervision — no verifier is consulted.
        """
        if self.judge_optimizer is None:
            return []
        losses: list[float] = []
        with self.ledger.training_scope(), self.ledger.access(
                Level.L1, Role.METHOD_WARMUP, "judge warm-up"):
            for r in range(n_rounds):
                ids, _ = self.stage_prompts(stage, n_per_round, seed=9000 + r)
                # demo_answer, not correct_token: the latter is the verifier's
                # view and is unreachable inside a training scope in a real
                # backend.  Using it here would perform an L3 read under an L1
                # grant, which the ledger cannot detect because the call never
                # passes through it.
                good = self.world.demo_answer(stage, ids)
                bad = (good + 1) % self.world.vocab
                self.judge_optimizer.zero_grad(set_to_none=True)
                loss = torch.zeros(())
                for ans, label in ((good, 1.0), (bad, 0.0)):
                    # judge_prob is backend-neutral: the toy returns the
                    # softmax mass on the answer token, a GenRM judge returns
                    # sigma(z_Yes - z_No).  Branching here would mean the
                    # warm-up trained two different objectives on two backends.
                    probs = self.model.judge_prob(ids, ans)          # (P,)
                    loss = loss + F.binary_cross_entropy(
                        probs.clamp(1e-6, 1 - 1e-6),
                        torch.full_like(probs, label))
                loss = loss / 2.0
                loss.backward()
                self.judge_optimizer.step()
                losses.append(float(loss.detach()))
        return losses

    def freeze_judge(self) -> None:
        """Freeze the current judge as the *protected* snapshot for E2 arm (ii)."""
        self.protected_judge = ModuleState.capture(self.model.judge_bank)

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------
    def reward_matrix(self, stage: int, prompt_ids: torch.Tensor,
                      answers: torch.Tensor, *, purpose: str
                      ) -> torch.Tensor:
        """Score ``answers`` for the run's reward source.

        GOLD is the only branch that reads L3; the ledger's exemption applies
        because callers open a gold-control scope.  Everything else uses the
        judge (current or frozen-protected) — L2 supervision.
        """
        src = self.plan.reward_source
        self.model.set_stage(stage)
        if src in (RewardSource.SELF, RewardSource.DEGENERATE_ZEROED):
            if self.plan.icl_judge:
                # Row 4: theta_J is never trained; the adaptation is an
                # in-context prefix, so the judge must see it at scoring time
                from ..methods.baselines import icl_context_prompts

                ctx_ids = icl_context_prompts(self.world, stage, prompt_ids,
                                              n_shots=self.icl_shots, seed=stage)
                return self.model.judge_score_matrix(prompt_ids, answers,
                                                     context_ids=ctx_ids)
            return self.model.judge_score_matrix(prompt_ids, answers)
        if src is RewardSource.PROTECTED:
            if self.protected_judge is None:
                raise RuntimeError("call freeze_judge() before using the protected arm")
            with self.protected_judge.context(self.model.judge_bank):
                return self.model.judge_score_matrix(prompt_ids, answers)
        if src is RewardSource.GOLD:
            self.ledger.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM,
                               f"gold reward arm:{purpose}")
            return self.model.gold_score_matrix(prompt_ids, answers)
        raise ValueError(src)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train_step(self, ctx: StageContext, step: int) -> dict:
        if self.plan.static_mix:
            return self._static_mix_step(ctx, step)
        G = self.cfg.train.group_size
        idx = int(torch.randint(0, len(ctx.prompts), (1,), generator=self.rng))
        pid = ctx.prompts[idx: idx + 1]

        self.model.set_stage(ctx.stage)
        with torch.no_grad():
            seqs, mask, old_lp = self.model.sample(pid, G, generator=self.rng)
        answers = seqs[:, -1].reshape(1, G)

        gold_control = self.is_gold_arm
        with self.ledger.training_scope(gold_control=gold_control):
            rewards = self.reward_matrix(ctx.stage, pid, answers,
                                         purpose=f"step{step}")
            adv = compute_advantages(to_np(rewards), cfg=self.adv_cfg)

            batch = RolloutBatch(
                task_ids=[ctx.task_ids[idx]],
                sequences=seqs.view(1, G, -1),
                attention_mask=mask.view(1, G, -1),
                old_logprob=old_lp.view(G, -1),
                prompt_len=pid.shape[1], sampled_at_step=step,
            )
            res = self.trainer.step_from_rollout(
                batch, to_tensor(adv.reshape(-1)).float()
            )

        return {
            "step": step,
            "prompt_idx": idx,
            "rewards": to_np(rewards),
            "adv": adv,
            "loss": res.loss,
            "grad_norm": res.grad_norm,
            "delta_norm": res.delta_norm,
            "gold_control": gold_control,
        }

    def supervised_step(self, ctx: StageContext, step: int,
                        *, icl: bool = False, self_distill: bool = False,
                        n_shots: int = 6) -> dict:
        """One off-policy supervised update on stage demonstrations (L1 only).

        Used by the non-RL baselines.  ``icl=True`` distils the model's own
        prediction made on a demonstration-conditioned context (Row 2b), which
        keeps the prompt budget aligned with GRPO without ever consulting gold.
        """
        from ..methods.baselines import demonstration_batch, icl_context_prompts

        n = min(16, self.cfg.data.demo_per_stage)
        ids, targets = demonstration_batch(self.world, ctx.stage, n,
                                           seed=step * 7919)
        self.model.set_stage(ctx.stage)
        teacher = None
        if self_distill and not icl:
            # Row 2a: distil the model's own softened prediction on the demos.
            with torch.no_grad():
                teacher = torch.softmax(self.model.actor_logits(ids) / 2.5, dim=-1)
        if icl:
            # Row 2b: distil the model's prediction *conditioned on demos*.
            # The toy actor is per-row, so the context must be passed as a
            # conditioning vector; concatenating rows would leave the teacher
            # exactly equal to the student (measured KL = 0).
            ctx_ids = icl_context_prompts(self.world, ctx.stage, ids, n_shots,
                                          seed=step)
            self.model.set_stage(ctx.stage)
            with torch.no_grad():
                teacher_logits = self.model.actor_logits(ids, context_ids=ctx_ids)
            teacher = torch.softmax(teacher_logits / 2.5, dim=-1)
        res = self.trainer.step_supervised(ids, targets, teacher_logprob=teacher)
        return {"step": step, "loss": res.loss, "delta_norm": res.delta_norm,
                "supervised": True, "icl": icl}

    def _static_mix_step(self, ctx: StageContext, step: int) -> dict:
        """Row 9: a batch whose rows come from several stages at once.

        Ordering is removed, so this is the non-continual reference that defines
        the Continual Gap.  The actor receives a per-row ``stage_ids`` vector;
        the judge is frozen (no stage-wise drift to measure).
        """
        stages = tuple(range(1, len(self.cfg.data.stages) + 1))
        G = self.cfg.train.group_size
        per = max(1, G // len(stages))
        n_stage = len(stages)

        pids, sids = [], []
        for st in stages:
            ids, _ = self.stage_prompts(st, per, seed=step * 31 + st)
            pids.append(ids)
            sids.append(torch.full((len(ids),), st - 1, dtype=torch.long))
        pid = torch.cat(pids, dim=0)
        stage_ids = torch.cat(sids, dim=0)

        with torch.no_grad():
            logits = self.model.actor_logits(pid, stage_ids=stage_ids)
            probs = torch.softmax(logits, dim=-1)
            draws = torch.multinomial(probs.repeat_interleave(per, dim=0), 1,
                                      generator=self.rng).squeeze(-1)
            answers = draws.view(len(pid), per)
            seqs = torch.cat([pid.unsqueeze(1).expand(-1, per, -1).reshape(-1, pid.shape[1]),
                              answers.reshape(-1, 1)], dim=1)
            mask = torch.zeros_like(seqs)
            mask[:, -1] = 1.0
            old_lp = self.model.logprob(seqs, mask, pid.shape[1])

        with self.ledger.training_scope():
            rewards = self.model.judge_score_matrix(pid, answers)
            adv = compute_advantages(to_np(rewards), cfg=self.adv_cfg)
            batch = RolloutBatch(
                task_ids=[f"mix{st}" for st in range(len(pid)) for _ in range(per)],
                sequences=seqs.view(len(pid), per, -1),
                attention_mask=mask.view(len(pid), per, -1),
                old_logprob=old_lp.view(len(pid) * per, -1),
                prompt_len=pid.shape[1], sampled_at_step=step,
            ).collapse()
            res = self.trainer.step_from_rollout(
                batch, to_tensor(adv.reshape(-1)).float())

        return {"step": step, "loss": res.loss, "delta_norm": res.delta_norm,
                "static_mix": True, "n_rows": int(len(pid))}

    # ------------------------------------------------------------------
    # measurement
    # ------------------------------------------------------------------
    def score_with_gold(self, stage: int, prompt_ids: torch.Tensor,
                        answers: torch.Tensor, *, role: str = Role.AUDIT_REPORTING
                        ) -> torch.Tensor:
        self.model.set_stage(stage)
        self.ledger.access(Level.L3, role, f"audit line @stage{stage}")
        return self.model.gold_score_matrix(prompt_ids, answers)

    def measure_onpolicy(self, ctx: StageContext, step: int,
                         n_prompts: int | None = None) -> Observation:
        """Line B: current-stage rollouts, scored by the live judge + gold audit."""
        n = n_prompts or min(32, len(ctx.prompts))
        pid = ctx.prompts[:n]
        G = self.cfg.train.group_size
        self.model.set_stage(ctx.stage)
        with torch.no_grad():
            seqs, _, _ = self.model.sample(pid, G, generator=self.rng)
        answers = seqs[:, -1].reshape(n, G)
        judge = to_np(self.model.judge_score_matrix(pid, answers))
        gold = to_np(self.score_with_gold(ctx.stage, pid, answers))
        sample = self.dual_line.measure(LINE_ONPOLICY, step, judge, gold)
        return Observation(step=step, line=LINE_ONPOLICY,
                           values=dict(sample.report.values),
                           n_informative=sample.report.n_informative,
                           n_degenerate=sample.report.n_degenerate)

    def measure_audit(self, ctx: StageContext, step: int,
                      answers: torch.Tensor | None = None) -> Observation:
        """Line A: frozen old-stage prompts and answers, scored by the live judge."""
        if ctx.audit_prompts is None:
            raise ValueError("context has no audit prompts")
        G = self.cfg.train.group_size
        if answers is None:
            self.model.set_stage(ctx.stage)
            with torch.no_grad():
                seqs, _, _ = self.model.sample(ctx.audit_prompts, G,
                                               generator=self.rng)
            answers = seqs[:, -1].reshape(len(ctx.audit_prompts), G)
        return self._measure_audit_fixed(ctx, step, answers, audit_stage=1)

    def _measure_audit_fixed(self, ctx: StageContext, step: int,
                             answers: torch.Tensor, audit_stage: int) -> Observation:
        """Score a *fixed* answer set on P_audit with the live judge (line A).

        The answer set is frozen so that consecutive measurements isolate judge
        drift; output-distribution shift is measured separately by the 2x2
        factorial, which re-samples ``Y_k``.
        """
        if ctx.audit_prompts is None:
            raise ValueError("context has no audit prompts")
        judge = to_np(self.model.judge_score_matrix(ctx.audit_prompts, answers))
        gold = to_np(self.score_with_gold(audit_stage, ctx.audit_prompts,
                                    answers))
        sample = self.dual_line.measure(LINE_AUDIT, step, judge, gold)
        return Observation(step=step, line=LINE_AUDIT,
                           values=dict(sample.report.values),
                           n_informative=sample.report.n_informative,
                           n_degenerate=sample.report.n_degenerate)


    # ------------------------------------------------------------------
    # 2x2 snapshot factorial
    # ------------------------------------------------------------------
    def capture_stage_start(self, ctx: StageContext, audit_stage: int
                           ) -> torch.Tensor:
        """Sample ``Y_0`` once at stage start and freeze the judge snapshot.

        Returns the frozen answer set so the *same* answers can be re-scored by
        the end-of-stage judge (that pairing is what makes ``E10 − E00`` a
        pure judge-drift term).
        """
        # save previous stage's end-judge *before* overwriting protected_judge;
        # this is θ_J^(t-1,end), needed for the E_w boundary term (H6)
        if self.protected_judge is not None:
            self.prev_stage_judge = self.protected_judge
        self.freeze_judge()
        self.model.set_stage(audit_stage)
        with torch.no_grad():
            seqs, _, _ = self.model.sample(ctx.audit_prompts,
                                           self.cfg.train.group_size,
                                           generator=self.rng)
        return seqs[:, -1].reshape(len(ctx.audit_prompts), self.cfg.train.group_size)

    def measure_factorial(self, ctx: StageContext, step: int,
                          y0_answers: torch.Tensor, audit_stage: int,
                          *, kind: str = "acc_rank",
                          n_resamples: int = 500,
                          canon_scores: tuple | None = None) -> dict:
        """Build the 2x2 table and return the attribution decomposition.

        ``θ_J^(t,0)`` is the protected snapshot (``E00``/``E01``);
        ``θ_J^(t,k)`` is the live judge (``E10``/``E11``).  ``Y_0`` is the
        stage-start answer set, ``Y_k`` the current one.  Gold labels are audit
        reads taken outside any training scope.
        """
        from ..diagnostics.factorial import SnapshotPair, factorial_attribution

        if self.protected_judge is None:
            raise RuntimeError("call capture_stage_start() before measuring")
        if ctx.audit_prompts is None:
            raise ValueError("context has no audit prompts")

        G = self.cfg.train.group_size
        self.model.set_stage(audit_stage)
        with torch.no_grad():
            seqs, _, _ = self.model.sample(ctx.audit_prompts, G, generator=self.rng)
        yk_answers = seqs[:, -1].reshape(len(ctx.audit_prompts), G)

        gold0 = to_np(self.score_with_gold(audit_stage, ctx.audit_prompts, y0_answers))
        goldk = to_np(self.score_with_gold(audit_stage, ctx.audit_prompts, yk_answers))

        # judge_end (live)
        end0 = to_np(self.model.judge_score_matrix(ctx.audit_prompts, y0_answers))
        endk = to_np(self.model.judge_score_matrix(ctx.audit_prompts, yk_answers))
        # judge_start (protected snapshot)
        with self.protected_judge.context(self.model.judge_bank):
            start0 = to_np(self.model.judge_score_matrix(ctx.audit_prompts, y0_answers))
            startk = to_np(self.model.judge_score_matrix(ctx.audit_prompts, yk_answers))

        pair = SnapshotPair(
            judge_start_on_y0=start0, judge_end_on_y0=end0,
            judge_start_on_yk=startk, judge_end_on_yk=endk,
            gold_on_y0=gold0, gold_on_yk=goldk,
            tau=self.cfg.train.degenerate_tau,
        )
        res = factorial_attribution(pair, kind=kind, n_resamples=n_resamples,
                                    seed=self.cfg.seed + step)
        out = res.as_dict()

        # E_w boundary term + H6 non-inferiority test.  Requires the previous
        # stage's end judge (θ_J^(t-1,end)) captured by capture_stage_start().
        # On the very first stage there is no previous stage, so we skip it.
        if self.prev_stage_judge is not None:
            from ..diagnostics.factorial import boundary_attribution
            from ..diagnostics.bootstrap import non_inferiority_test

            with self.prev_stage_judge.context(self.model.judge_bank):
                prev_on_old = to_np(self.model.judge_score_matrix(
                    ctx.audit_prompts, y0_answers))

            bc = boundary_attribution(
                prev_on_old,
                start0,       # θ_J^(t,0) already computed above
                gold0,
                pair,
                kind=kind,
                n_resamples=n_resamples,
                seed=self.cfg.seed + step + 1,
            )
            # H6 non-inferiority test: reject H0 (E_w ≤ −δ_NI) when lo > −margin.
            # paired_diff is per-prompt: error(θ_J^(t,0)) − error(θ_J^(t-1,end))
            # which equals metric_now − metric_prev (metric = 1 − error).
            from ..diagnostics.fidelity import per_prompt_stats
            ps_now = per_prompt_stats(start0, gold0, tau=pair.tau)
            ps_prev = per_prompt_stats(prev_on_old, gold0, tau=pair.tau)
            _now_arr = getattr(ps_now, kind.replace("-", "_"), ps_now.acc_rank)
            _prev_arr = getattr(ps_prev, kind.replace("-", "_"), ps_prev.acc_rank)
            paired_diff = _now_arr - _prev_arr
            ni = non_inferiority_test(
                paired_diff,
                margin=self.cfg.exp.delta_ni_pt / 100.0,
                level=self.cfg.exp.ci_level,
            )
            out["boundary_chain"] = bc.as_dict()
            out["h6_non_inferiority"] = ni.as_dict()

        # ΔÂ_canon: optional advantage-sensitivity probe (H1 supplementary).
        # Requires pre-computed judge scores on a rewritten answer set; the
        # text rewriting is external (domain / backend dependent), so the
        # caller passes them in when available.  When absent, the key is
        # omitted so downstream code can check presence rather than nan.
        if canon_scores is not None:
            from ..diagnostics.canonicalization import compute_canon
            orig_sc, rew_sc = canon_scores
            canon_rep = compute_canon(np.asarray(orig_sc), np.asarray(rew_sc))
            out["canon"] = canon_rep.as_dict()
        return out

    # ------------------------------------------------------------------
    # snapshots
    # ------------------------------------------------------------------
    def snapshot(self, tag: str) -> TrainingState:
        return self.trainer.snapshot(tag=tag)

    def restore(self, state: TrainingState) -> None:
        self.trainer.restore(state)
