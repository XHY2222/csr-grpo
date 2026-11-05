"""RLER-faithful and the judge-ensemble variant (Row 5a / 5a').

RLER's mechanism (arXiv:2510.08977 §4.1), as implemented here:

* maintain ``M`` **generation sub-policies** (default ``M = 2``) that share the
  total rollout budget: each produces ``G / M`` rollouts, so the sampling cost
  matches a single-policy GRPO run with group size ``G``;
* **aggregate the answer distributions** of the sub-policies into one effective
  reward per answer;
* apply **adaptive soft/hard reward interpolation**: when the sub-policies
  agree, use the hard (discrete) signal, when they disagree, fall back to the
  soft (probability-averaged) signal with a disagreement-dependent weight;
* perform **disagreement-aware selection** of which rollouts enter the batch.

What this deliberately is *not*: a multi-judge LoRA ensemble.  The plan
explicitly separates "several sub-policies generating rollouts" from "one actor
scored by K judge adapters"; ``JudgeEnsembleMethod`` below is the latter and is
reported as the paper's own ablation, never as RLER.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.schema import Config
from ..methods.lora import AdapterSpec
from ..rewards.advantage import AdvMode
from .base import BaseMethod, MethodPlan


# ---------------------------------------------------------------------------
# aggregation maths (pure, testable)
# ---------------------------------------------------------------------------


def aggregate_distributions(subpolicy_probs: np.ndarray,
                            weights: np.ndarray | None = None) -> np.ndarray:
    """Aggregate ``(M, V)`` sub-policy distributions into one ``(V,)``.

    Log-linear averaging would be the naive choice; RLER aggregates the answer
    *distributions*, so a weighted arithmetic mean is the faithful version.
    """
    p = np.asarray(subpolicy_probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("expected (M, V)")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each sub-policy row must sum to 1")
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return (w[:, None] * p).sum(axis=0)


def disagreement(p: np.ndarray) -> float:
    """Jensen-Shannon divergence between sub-policy distributions, in [0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    m = p.mean(axis=0)
    def _kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    js = 0.5 * (_kl(p[0], m) + _kl(p[1], m)) if len(p) == 2 else \
        float(np.mean([_kl(row, m) for row in p]))
    return float(np.clip(js / np.log(2), 0.0, 1.0))


@dataclass
class AdaptiveInterpolation:
    """Soft/hard reward mixing with a disagreement-dependent weight.

    ``lam = 0`` → hard signal only; ``lam = 1`` → fully soft.  The paper's
    rationale is that hard labels are sharper when sub-policies agree, while
    soft labels carry the "how confident" information when they do not.
    """

    k: float = 4.0        # steepness of the agreement gate
    midpoint: float = 0.25  # disagreement at which soft and hard weigh equally

    def weight(self, js: float) -> float:
        return float(1.0 / (1.0 + np.exp(-self.k * (js - self.midpoint))))

    def mix_with(self, soft: np.ndarray, hard: np.ndarray, js: float) -> np.ndarray:
        lam = self.weight(js)
        return (1.0 - lam) * np.asarray(hard, dtype=np.float64) + \
            lam * np.asarray(soft, dtype=np.float64)


def rler_rewards(answer_ids: np.ndarray, agg_probs: np.ndarray, *,
                 interp: AdaptiveInterpolation, js: float) -> np.ndarray:
    """Reward each sampled answer under the aggregated distribution.

    ``soft``  = aggregated probability of the sampled answer;
    ``hard``  = whether the answer is the argmax of the aggregated distribution.
    """
    a = np.asarray(answer_ids, dtype=np.int64)
    soft = agg_probs[a]
    best = int(np.argmax(agg_probs))
    hard = (a == best).astype(np.float64)
    return interp.mix_with(soft, hard, js)


def disagreement_aware_select(logprob_sub: np.ndarray, keep_fraction: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Keep a disagreement-weighted subset of rollouts (indices).

    ``logprob_sub`` is ``(N, M)``: the log prob each sub-policy assigns to a
    sampled answer.  High cross-sub-policy variance marks informative samples.
    """
    lp = np.asarray(logprob_sub, dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError("expected (N, M)")
    spread = lp.std(axis=1)
    if spread.max() == 0:
        idx = rng.permutation(len(lp))
    else:
        # stochastic rank-based keep: probability proportional to spread
        p = spread / spread.sum()
        n_keep = max(1, int(round(keep_fraction * len(lp))))
        idx = rng.choice(len(lp), size=n_keep, replace=False, p=p)
    return np.sort(idx)


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


@dataclass
class SharedGRPOMethod(BaseMethod):
    """Row 3: one shared adapter, judge updated inside the stage."""

    name: str = "shared_grpo"
    spec: AdapterSpec = field(default_factory=AdapterSpec.shared)

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=True,
            notes=("shared adapter r_S>0: judge drift E_10-E_00 is measurable",),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "SharedGRPOMethod":
        # A shared baseline must actually have a shared branch.  Configs written
        # for Dual-Adapter carry r_shared=0, so fall back to the private rank
        # rather than silently producing a method with no trainable adapters.
        r_shared = cfg.model.r_shared or cfg.model.r_actor
        return cls(spec=AdapterSpec.shared(
            r=r_shared,
            alpha=float(cfg.model.lora_alpha),
            r_private=cfg.model.r_actor,
        ))


@dataclass
class RLERFaithfulMethod(BaseMethod):
    name: str = "rler_faithful"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    n_subpolicies: int = 2
    group_size: int = 8
    keep_fraction: float = 0.75
    interp: AdaptiveInterpolation = field(default_factory=AdaptiveInterpolation)

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=True,
            n_subpolicies=self.n_subpolicies,
            notes=(
                f"M={self.n_subpolicies} generation sub-policies, "
                f"{self.group_size // self.n_subpolicies} rollouts each "
                f"(matched budget G={self.group_size})",
                "aggregate answer distributions + adaptive soft/hard interpolation",
                "not to be confused with a multi-judge LoRA ensemble",
            ),
        )

    @property
    def per_policy_group(self) -> int:
        if self.group_size % self.n_subpolicies:
            raise ValueError("group_size must be divisible by M")
        return self.group_size // self.n_subpolicies

    @classmethod
    def from_config(cls, cfg: Config) -> "RLERFaithfulMethod":
        return cls(
            spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                          alpha=float(cfg.model.lora_alpha)),
            n_subpolicies=2,
            group_size=cfg.train.group_size,
            interp=AdaptiveInterpolation(),
        )


@dataclass
class JudgeEnsembleMethod(BaseMethod):
    """Paper's own ablation: single actor, K judge adapters voting."""

    name: str = "judge_lora_ensemble"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    k: int = 3

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=False,
            judge_ensemble_k=self.k,
            notes=(f"K={self.k} judge adapters score the same rollouts (our ablation, "
                   "not the RLER mechanism)",),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "JudgeEnsembleMethod":
        return cls(spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                                 alpha=float(cfg.model.lora_alpha)),
                   k=cfg.judge.ensemble_k)


def ensemble_scores(score_list: list[np.ndarray],
                    method: str = "mean") -> np.ndarray:
    """Vote/mean across K judge adapters on identical rollouts."""
    if not score_list:
        raise ValueError("no judges to ensemble")
    stack = np.stack([np.asarray(s) for s in score_list], axis=0)
    if method == "mean":
        return stack.mean(axis=0)
    if method == "rank":
        from ..rewards.advantage import rank_transform
        flat = stack.reshape(-1, stack.shape[-1])
        return rank_transform(flat).reshape(stack.shape).mean(axis=0)
    raise ValueError(method)
