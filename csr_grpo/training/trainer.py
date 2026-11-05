"""The GRPO step, and the only place a training scope is opened.

Design points that the paper's protocol depends on:

* ``step_from_rollout`` takes an *already frozen* rollout batch and a reward
  matrix.  Protocol A (E2) calls exactly this, so a diagnostic branch and a
  training step are the same code path — no "diagnostic reimplementation" drift.
* The entire optimisation body runs inside ``ledger.training_scope()``.  Any L3
  access from a reward provider called here raises ``SupervisionViolation``.
* ``Δθ`` is read as ``θ_after − θ_before`` after the *real* optimizer step, so
  Adam momentum and gradient clipping are included by construction
  (paper §F2: replace the basis-dependent ``⟨g, U_old⟩`` with ΔL_old ≈ ∇L_oldᵀ Δθ).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import torch
import torch.nn.functional as F

from ..supervision.ledger import Level, Role, SupervisionLedger
from .state import TrainingState


class Policy(Protocol):
    """Minimal interface shared by the toy and HF backends."""

    def sample(self, task_ids: Sequence[str], group_size: int,
               temperature: float = 1.0, generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(sequences, attention_mask, old_logprob)``."""

    def logprob(self, sequences: torch.Tensor, attention_mask: torch.Tensor,
                prompt_len: int) -> torch.Tensor:
        """Per-token log probs of the completion, shape ``(N, T-1)``."""

    def named_parameters(self): ...
    def parameters(self): ...


@dataclass
class StepResult:
    loss: float
    kl: float
    grad_norm: float
    clip_fraction: float
    adv_mean: float
    adv_std: float
    n_tokens: int
    delta_norm: float
    delta_signature: dict[str, torch.Tensor] = field(default_factory=dict)


def grpo_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    ref_logprob: torch.Tensor | None = None,
    kl_coef: float = 0.0,
    ref_kl_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level GRPO objective.

    ``advantages`` is ``(N,)`` (already group-standardised) and is broadcast
    over the completion.  Returns the scalar loss (to minimise) plus diagnostics.
    """
    if new_logprob.shape != old_logprob.shape:
        raise ValueError("logprob shape mismatch")
    n = new_logprob.shape[0]
    if advantages.numel() != n:
        raise ValueError(f"expected {n} advantages, got {advantages.numel()}")

    adv = advantages.to(new_logprob.device).detach()[:, None]
    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token = -torch.min(unclipped, clipped)

    if attention_mask is not None:
        mask = attention_mask[:, 1:].to(per_token.device).float()
        denom = mask.sum().clamp(min=1.0)
        loss = (per_token * mask).sum() / denom
        clip_frac = (((ratio - 1.0).abs() > clip_eps).float() * mask).sum() / denom
        n_tokens = float(mask.sum())
    else:
        loss = per_token.mean()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
        n_tokens = float(per_token.numel())

    kl_val = 0.0
    if ref_logprob is not None and kl_coef != 0.0:
        log_r = ref_logprob.detach() - new_logprob
        kl = (torch.exp(log_r) - log_r - 1.0)
        # ref_kl_clip caps outlier KL values so a very divergent token does not
        # swamp the loss signal; clipping is optional (None = no clip).
        if ref_kl_clip is not None:
            kl = torch.clamp(kl, max=ref_kl_clip)
        if attention_mask is not None:
            kl = (kl * mask).sum() / denom
        else:
            kl = kl.mean()
        loss = loss + kl_coef * kl
        kl_val = float(kl.detach())

    return loss, {
        "kl": kl_val,
        "clip_fraction": float(clip_frac.detach()),
        "n_tokens": int(n_tokens),
    }


class GRPOTrainer:
    """Owns the optimizer, the step counter and the ledger scope."""

    def __init__(self, policy: Policy, optimizer: torch.optim.Optimizer,
                 ledger: SupervisionLedger, *,
                 grad_clip: float = 1.0, kl_coef: float = 0.0,
                 clip_eps: float = 0.2, device: str = "cpu",
                 ref_kl_clip: float | None = None):
        self.policy = policy
        self.optimizer = optimizer
        self.ledger = ledger
        self.grad_clip = grad_clip
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.ref_kl_clip = ref_kl_clip
        self.device = device
        self.step_count = 0
        self._ref_state: TrainingState | None = None
        self.history: list[StepResult] = []

    @property
    def _optimized_params(self) -> list[torch.nn.Parameter]:
        """Parameters this optimizer actually writes to.

        Clipping must be restricted to these.  ``module.parameters()`` also
        covers the judge adapters, whose ``.grad`` survives
        ``optimizer.zero_grad()`` (the actor optimizer does not own them).  The
        consequence was measured in protocol A: an arm whose advantages were
        all zero produced ``grad_norm = 0.362`` -- entirely stale judge
        gradients -- and that number then scaled the *actor* update, so the
        step size depended on the judge's history rather than on the reward.
        """
        out: list[torch.nn.Parameter] = []
        for group in self.optimizer.param_groups:
            out.extend(group["params"])
        return out

    # -- reference model ----------------------------------------------------
    def freeze_reference(self, module: torch.nn.Module) -> None:
        self._ref_state = TrainingState.capture(module, self.optimizer,
                                                step=self.step_count,
                                                role="reference")

    def _ref_logprob(self, sequences, attention_mask, prompt_len) -> torch.Tensor | None:
        if self._ref_state is None or self.kl_coef == 0.0:
            return None
        module = self.policy
        live = TrainingState.capture(module, self.optimizer, step=self.step_count)
        self._ref_state.restore(module, self.optimizer, restore_rng=False)
        with torch.no_grad():
            ref = module.logprob(sequences, attention_mask, prompt_len)
        live.restore(module, self.optimizer, restore_rng=False)
        return ref.detach()

    # -- the step -----------------------------------------------------------
    def step_supervised(self, prompt_ids: torch.Tensor,
                        target_ids: torch.Tensor, *,
                        teacher_logprob: torch.Tensor | None = None,
                        role: str = Role.METHOD_TRAIN,
                        record_delta: bool = True) -> StepResult:
        """Supervised (SFT / SDFT) update on demonstrations.

        SFT minimises the NLL of the target answer.  SDFT instead distils the
        model's *own* distribution on an ICL context, so ``teacher_logprob``
        supplies the teacher targets and the loss becomes a KL.  Both are L1
        supervision (stage-initial demos), so the same training scope and the
        same gold ban apply — a supervised baseline cannot become a gold oracle
        by accident.
        """
        module = self.policy
        before = TrainingState.capture(module, self.optimizer, step=self.step_count)

        with self.ledger.training_scope(), self.ledger.access(
                Level.L2, role, purpose="supervised step"):
            self.optimizer.zero_grad(set_to_none=True)
            logits = module.actor_logits(prompt_ids)
            logp = torch.log_softmax(logits, dim=-1)
            if teacher_logprob is None:
                # plain SFT: NLL of the demonstrated answer
                loss = -logp.gather(1, target_ids[:, None]).mean()
                kl = 0.0
            else:
                # SDFT: forward KL from the self-distilled teacher
                teacher = teacher_logprob.to(logp.device)
                loss = torch.sum(teacher * (teacher.log() - logp), dim=-1).mean()
                kl = float(loss.detach())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._optimized_params, self.grad_clip)
            self.optimizer.step()

        self.step_count += 1
        delta = before.param_delta(module) if record_delta else {}
        result = StepResult(
            loss=float(loss.detach()), kl=kl, grad_norm=float(grad_norm),
            clip_fraction=0.0, adv_mean=0.0, adv_std=0.0,
            n_tokens=int(prompt_ids.shape[0]),
            delta_norm=float(torch.sqrt(sum((d ** 2).sum() for d in delta.values()))) if delta else 0.0,
            delta_signature=delta,
        )
        self.history.append(result)
        return result

    def step_from_rollout(self, rollout, advantages: torch.Tensor, *,
                          role: str = Role.METHOD_TRAIN,
                          record_delta: bool = True) -> StepResult:
        """One optimiser step over a frozen batch.

        Args:
            rollout: ``RolloutBatch`` (already collapsed to ``(P*G, T)``).
            advantages: ``(P*G,)`` per-sample advantages.
        """
        module = self.policy
        # Protocol A hands every branch the same frozen rollout; collapse here
        # so a caller cannot accidentally pass a (1, G, T) batch that would be
        # read as an N x 5 matrix.
        if rollout.sequences.ndim == 3:
            rollout = rollout.collapse()
        before = TrainingState.capture(module, self.optimizer, step=self.step_count)

        with self.ledger.training_scope(), self.ledger.access(
                Level.L2, role, purpose="grpo step"):
            self.optimizer.zero_grad(set_to_none=True)
            lp = module.logprob(rollout.sequences, rollout.attention_mask,
                                rollout.prompt_len)
            ref = self._ref_logprob(rollout.sequences, rollout.attention_mask,
                                    rollout.prompt_len)
            loss, diag = grpo_loss(
                lp, rollout.old_logprob, advantages,
                attention_mask=rollout.attention_mask,
                clip_eps=self.clip_eps,
                ref_logprob=ref,
                kl_coef=self.kl_coef,
                ref_kl_clip=self.ref_kl_clip,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._optimized_params, self.grad_clip)
            self.optimizer.step()

        self.step_count += 1
        adv = advantages.detach()
        delta = before.param_delta(module) if record_delta else {}

        result = StepResult(
            loss=float(loss.detach()),
            kl=diag["kl"],
            grad_norm=float(grad_norm),
            clip_fraction=diag["clip_fraction"],
            adv_mean=float(adv.mean()),
            adv_std=float(adv.std(unbiased=False)),
            n_tokens=diag["n_tokens"],
            delta_norm=float(torch.sqrt(sum((d ** 2).sum() for d in delta.values()))) if delta else 0.0,
            delta_signature=delta,
        )
        self.history.append(result)
        return result

    # -- snapshots ----------------------------------------------------------
    def snapshot(self, **tags) -> TrainingState:
        return TrainingState.capture(self.policy, self.optimizer,
                                     step=self.step_count, **tags)

    def restore(self, state: TrainingState) -> None:
        state.restore(self.policy, self.optimizer)
        self.step_count = state.step

    def state_dict(self) -> dict:
        return {
            "step": self.step_count,
            "history": [r.__dict__.copy() for r in self.history],
        }
