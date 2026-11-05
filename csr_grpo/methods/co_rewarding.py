"""Frontier baselines: Meta-Rewarding (5b), online judge co-training (5c),
M-GRPO-adapted (6).

These are the rows the plan calls "决定性基线" — they test whether mechanisms
designed for *single-task* self-reward stabilisation can withstand cross-stage
switching.  Each is annotated with what it keeps and what it changes relative to
the cited work, because the plan insists on that distinction (§文献谱系 ①).

======  ===============================================  =====================
Row     mechanism kept                                     adapted here
======  ===============================================  =====================
5b      Meta-Judge scores its own judgements, preference   none: single-stage
        pairs train the Judge (2407.19594)                 procedure as written
5c      every RL step also updates the Judge on historical ours: labelled as an
        + current contrast pairs                           adaptation, not a paper
6       slow-teacher EMA (τ=0.995) + low-entropy filter    ours: EMA *judge*
        (Co-rewarding, M-GRPO)                             replaces the momentum
                                                           policy vote
======  ===============================================  =====================
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..config.schema import Config
from ..methods.lora import AdapterSpec, LowRankDelta
from ..rewards.advantage import AdvMode
from .base import BaseMethod, MethodPlan


# ===========================================================================
# Row 5b — Meta-Rewarding
# ===========================================================================


def meta_preference_pairs(scores: np.ndarray, meta_scores: np.ndarray
                          ) -> list[tuple[int, int]]:
    """Turn meta-judge scores into (winner, loser) index pairs within a group.

    The mechanism (2407.19594): the model judges its own judgements.  Given the
    judge's scores ``s`` and the meta-judge's scores ``m`` over the same answers,
    a pair is informative only when the meta-judge *disagrees* with the judge —
    agreeing pairs carry no learning signal for the judge.  Ties are dropped
    rather than broken arbitrarily, so the pair count is reported alongside.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = np.asarray(meta_scores, dtype=np.float64).reshape(-1)
    if s.shape != m.shape:
        raise ValueError("judge and meta-judge score vectors must align")
    pairs: list[tuple[int, int]] = []
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            if m[i] == m[j]:
                continue
            # the meta-judge decides which of the two judgements is better
            winner = i if m[i] > m[j] else j
            loser = j if winner == i else i
            # only keep it if the judge itself was *wrong* about the ordering
            if s[winner] < s[loser]:
                pairs.append((winner, loser))
    return pairs


def meta_preference_loss(scores: torch.Tensor,
                         pairs: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Bradley-Terry style loss that pushes the judge toward meta preferences."""
    if not pairs:
        return torch.zeros((), device=scores.device, requires_grad=True)
    w = torch.tensor([p[0] for p in pairs], device=scores.device)
    l = torch.tensor([p[1] for p in pairs], device=scores.device)
    diff = scores[w] - scores[l]
    return -F.logsigmoid(diff).mean()


@dataclass
class MetaRewardingMethod(BaseMethod):
    """Row 5b: judge is trained on its own meta-judgements, 2 rounds per stage."""

    name: str = "meta_rewarding"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    rounds_per_stage: int = 2
    group_size: int = 8

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=True,
            notes=(
                f"Meta-Judge -> Judge preference training, {self.rounds_per_stage} "
                "rounds per stage (arXiv:2407.19594 mechanism)",
                "keeps the meta-judge step unmodified; tests whether "
                "self-calibration survives cross-stage switching",
            ),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "MetaRewardingMethod":
        return cls(
            spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                          alpha=float(cfg.model.lora_alpha)),
            rounds_per_stage=2,
            group_size=cfg.train.group_size,
        )


# ===========================================================================
# Row 5c — online judge co-training + replay (our adaptation, clearly labelled)
# ===========================================================================


@dataclass
class OnlineJudgeCoTrainingMethod(BaseMethod):
    """Row 5c: judge gradient every RL step, on history + current pairs.

    This is *our* adaptation, not a published method — the plan requires the
    label.  It isolates one variable against Row 8: does updating the judge
    every step with historical replay beat freezing it inside the stage?
    """

    name: str = "online_judge_cotraining"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    replay_ratio: float = 1.0
    judge_lr_scale: float = 1.0

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=True,
            replay=True,
            notes=(
                "OUR ADAPTATION (not a published method): per-step judge "
                f"gradient on history+current contrast pairs at 1:{self.replay_ratio:g}",
                "answers 'is freezing the judge better than co-training it'",
            ),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "OnlineJudgeCoTrainingMethod":
        return cls(
            spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                          alpha=float(cfg.model.lora_alpha)),
            replay_ratio=float(cfg.train.replay_ratio),
        )


# ===========================================================================
# Row 6 — M-GRPO adapted (EMA judge + entropy filter)
# ===========================================================================


@dataclass
class EMAJudge:
    """Slow-moving copy of the judge's adapter weights.

    ``τ = 0.995`` means the teacher moves ~200× slower than the student, which
    is the Co-rewarding stabilisation idea.  We keep it as an explicit object
    rather than an optimizer trick so the effective lag is inspectable.
    """

    tau: float = 0.995
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)
    initialized: bool = False

    def update(self, params: dict[str, torch.Tensor]) -> None:
        if not self.initialized:
            self.shadow = {k: v.detach().clone() for k, v in params.items()}
            self.initialized = True
            return
        for k, v in params.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.tau).add_(v.detach(), alpha=1.0 - self.tau)

    def lens(self, params: dict[str, torch.Tensor]) -> float:
        """L2 distance between student and teacher — a drift read-out."""
        if not self.shadow:
            return float("nan")
        return float(torch.sqrt(sum(((params[k] - self.shadow[k]) ** 2).sum()
                                    for k in self.shadow if k in params)))


def entropy_filter_mask(logprobs: torch.Tensor,
                        keep_fraction: float = 0.5) -> torch.Tensor:
    """Keep the lowest-entropy trajectories (M-GRPO's filter).

    ``logprobs`` is ``(N, T)`` completion log probs.  Trajectory entropy is the
    mean per-token entropy proxy ``−log p``; the crop keeps the most confident
    ``keep_fraction`` of the batch, which is what removes noisy advantage.
    """
    if logprobs.dim() != 2:
        raise ValueError("expected (N, T) log probs")
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0,1]")
    # ``score`` is the mean per-token NEGATIVE log prob, so a confident
    # trajectory has a SMALL score.  Ascending order keeps the confident tail;
    # sorting descending here would retain exactly the noisy trajectories the
    # filter exists to remove.
    score = -logprobs.mean(dim=1)              # higher score = higher entropy
    n_keep = max(1, int(round(keep_fraction * score.numel())))
    order = torch.argsort(score, descending=False, stable=True)
    keep = torch.zeros_like(score, dtype=torch.bool)
    keep[order[:n_keep]] = True
    return keep


@dataclass
class MGRPOAdaptedMethod(BaseMethod):
    """Row 6: slow-teacher EMA judge + entropy filtering."""

    name: str = "mgrpo_adapted"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    tau: float = 0.995
    keep_fraction: float = 0.5
    ema: EMAJudge | None = None

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=True,
            notes=(
                f"OUR ADAPTATION: EMA judge (τ={self.tau}) + low-entropy filter "
                f"(keep {self.keep_fraction:.0%})",
                "kept from Co-rewarding/M-GRPO: slow teacher, entropy filter; "
                "changed: EMA judge replaces the momentum-policy vote",
            ),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "MGRPOAdaptedMethod":
        tau = cfg.judge.ema_tau if cfg.judge.ema_tau is not None else 0.995
        return cls(
            spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                          alpha=float(cfg.model.lora_alpha)),
            tau=float(tau),
        )

    def ema_judge(self) -> EMAJudge:
        if self.ema is None:
            self.ema = EMAJudge(tau=self.tau)
        return self.ema


# ===========================================================================
# entropy-filtered advantage assembly (used by Row 6 and the RLER variant)
# ===========================================================================


def apply_entropy_filter(advantages: np.ndarray, keep: torch.Tensor) -> np.ndarray:
    """Zero the advantages of filtered-out trajectories.

    Zeroing (rather than dropping rows) keeps the batch shape stable for the
    optimizer, and is equivalent because a zero advantage contributes nothing to
    ``grpo_loss``.
    """
    from ..compat import to_np

    a = np.array(advantages, dtype=np.float64, copy=True)
    # ``.numpy()`` is unavailable on this host (torch built against NumPy 1.x,
    # runtime is 2.2.6), so route through compat like every other bridge.
    mask = to_np(keep).astype(bool)
    if mask.shape[-1] != a.shape[0]:
        raise ValueError(
            f"mask covers {mask.shape[-1]} rows but advantages have {a.shape[0]}"
        )
    a[~mask] = 0.0
    return a
