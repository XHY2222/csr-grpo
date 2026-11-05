"""Reward sources for E2 and for the method baselines.

The four E2 arms (paper §E2) are:

    (i)   SELF               current judge θ_J^(t,k)
    (ii)  PROTECTED          frozen stage-start judge θ_J^(t,0) (+ replay ctx)
    (iii) GOLD               diagnostic gold verifier — L3, diagnostic only
    (iv)  DEGENERATE_ZEROED  current judge, advantage forced to 0 on groups the
                             judge believes are all-good / all-bad

A ``RewardProvider`` is a callable producing a ``(P, G)`` score matrix.  The
gold provider is the *only* one that touches L3 and it routes through the
ledger with the diagnostic role, so it raises inside a training scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np

from ..supervision.ledger import Level, Role, SupervisionLedger
from .advantage import AdvantageConfig, compute_advantages


class RewardSource(str, Enum):
    SELF = "self"
    PROTECTED = "protected"
    GOLD = "gold"
    DEGENERATE_ZEROED = "degenerate_zeroed"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


class RewardProvider(Protocol):
    source: RewardSource

    def score(self, task_ids: Sequence[str],
              answers: Sequence[Sequence[str]]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# judge-backed providers
# ---------------------------------------------------------------------------


@dataclass
class JudgeRewardProvider:
    """Scores answers with a judge callable (already temperature-0 logit)."""

    judge: Callable[[str, Sequence[str]], Sequence[float]]
    source: RewardSource = RewardSource.SELF
    name: str = "judge"

    def score(self, task_ids: Sequence[str],
              answers: Sequence[Sequence[str]]) -> np.ndarray:
        rows = [np.asarray(self.judge(t, a), dtype=np.float64)
                for t, a in zip(task_ids, answers)]
        out = np.vstack(rows)
        if not np.all(np.isfinite(out)):
            raise ValueError(f"judge {self.name} produced non-finite scores")
        return out


@dataclass
class GoldRewardProvider:
    """Diagnostic gold reward — L3, admissible only outside training scopes."""

    verifier: object                      # has .verify(task_id, answers, purpose)
    ledger: SupervisionLedger
    source: RewardSource = RewardSource.GOLD
    name: str = "gold"

    def score(self, task_ids: Sequence[str],
              answers: Sequence[Sequence[str]]) -> np.ndarray:
        # Explicit grant with the diagnostic role; the ledger raises if we are
        # inside a training scope, which is the whole point.
        self.ledger.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM,
                           f"{self.name}:E2-arm-(iii)")
        rows = [np.asarray(self.verifier.verify(t, a, purpose="e2_gold_arm"),
                           dtype=np.float64)
                for t, a in zip(task_ids, answers)]
        return np.vstack(rows)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


@dataclass
class RewardStack:
    """Everything one branch needs to turn rollouts into advantages."""

    provider: RewardProvider
    adv_cfg: AdvantageConfig
    ledger: SupervisionLedger
    gold_labels: np.ndarray | None = None   # only for degenerate mask in arm (iv)

    def advantages(self, task_ids: Sequence[str],
                   answers: Sequence[Sequence[str]]) -> tuple[np.ndarray, np.ndarray]:
        rewards = self.provider.score(task_ids, answers)
        adv = compute_advantages(
            rewards,
            cfg=self.adv_cfg,
            gold=self.gold_labels,
            ledger=self.ledger,
            mask_scores=rewards,
        )
        return rewards, adv


def build_arms(
    *,
    judge_current: Callable,
    judge_protected: Callable,
    gold_verifier: object,
    ledger: SupervisionLedger,
    adv_cfg: AdvantageConfig,
    arms: Sequence[str] = ("self", "protected", "gold", "degenerate_zeroed"),
) -> dict[RewardSource, RewardStack]:
    """Construct the E2 reward-source arms.

    Arm (iv) shares arm (i)'s provider and only flips ``degenerate_zero``; its
    advantage config is derived rather than passed in so the two branches can
    never drift apart accidentally.
    """
    stacks: dict[RewardSource, RewardStack] = {}
    for arm in arms:
        src = RewardSource(arm)
        if src is RewardSource.SELF:
            stacks[src] = RewardStack(
                JudgeRewardProvider(judge_current, src, "judge_current"),
                adv_cfg, ledger)
        elif src is RewardSource.PROTECTED:
            stacks[src] = RewardStack(
                JudgeRewardProvider(judge_protected, src, "judge_protected"),
                adv_cfg, ledger)
        elif src is RewardSource.GOLD:
            stacks[src] = RewardStack(
                GoldRewardProvider(gold_verifier, ledger),
                AdvantageConfig(mode=adv_cfg.mode, eps=adv_cfg.eps,
                                degenerate_zero=adv_cfg.degenerate_zero,
                                degenerate_source=adv_cfg.degenerate_source,
                                clip=adv_cfg.clip),
                ledger)
        elif src is RewardSource.DEGENERATE_ZEROED:
            cfg = AdvantageConfig(
                mode=adv_cfg.mode, eps=adv_cfg.eps,
                degenerate_zero=True,
                degenerate_source=adv_cfg.degenerate_source,
                clip=adv_cfg.clip,
            )
            stacks[src] = RewardStack(
                JudgeRewardProvider(judge_current, src, "judge_current+dz"),
                cfg, ledger)
        else:  # pragma: no cover
            raise ValueError(arm)
    return stacks


def assert_no_gold_in_training(stacks: dict[RewardSource, RewardStack],
                               ledger: SupervisionLedger) -> None:
    """Guard called by training loops that share a stack dictionary."""
    if ledger.in_training_scope and RewardSource.GOLD in stacks:
        # Building is fine; scoring would raise. We check intent here so the
        # error points at the configuration rather than at a random tensor.
        raise PermissionError(
            "a gold reward arm is registered while inside a training scope; "
            "gold arms belong to E2 diagnostics only"
        )
