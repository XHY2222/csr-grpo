"""Advantage-fidelity metrics on one measurement line.

A measurement line is a matrix of scores ``S`` of shape ``(P, K)``: ``P``
independent prompts, ``K`` candidate answers each, with a gold correctness
vector ``G`` of the same shape.  Both the blind audit line (fixed ``P_audit``
snapshots) and the on-policy line (live rollouts) reduce to this form, which is
why every statistic below is written once and reused.

Definitions (matching the paper, §F1)
-------------------------------------
Group advantage, for scores ``r`` in a group of size ``K``::

    Â_i = (r_i − μ_r) / (σ_r + ε)

``Acc_rank`` — within-group pairwise ranking accuracy restricted to
*gold-discordant* pairs, i.e. pairs ``(i, j)`` with ``g_i > g_j``.  A correct
gold set has no discordant pair, so degenerate groups (all correct / all wrong)
have no denominator: they are excluded, never filled with 0.

``Sign_agree`` — over *informative mixed* groups (``0 < #correct < K``),
``P(sign(Â_J) == sign(Â_gold))`` with a deadband so that numerically-zero
advantages count as 0 rather than ±.

``ρ_Adv`` — per-group Pearson correlation between ``Â_J`` and ``Â_gold``,
aggregated across groups through Fisher z.  Under group-standardised
advantages, ``MSE(Â_J, Â_gold) = 2(1 − ρ_Adv)`` identically, so MSE is not
reported separately (it is asserted in tests as a self-check).

``Pseudo-contrast rate`` — on degenerate gold groups (``σ_gold = 0``), the
fraction where the judge still manufactures a spread
``max_i r_J,i − min_i r_J,i > τ``.  We also report ``σ_J | degenerate``.

Everything is computed per prompt and returned as flat arrays so that the
cluster bootstrap can resample *prompts* (not answers) and recompute the
statistic on the resample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

MetricName = Literal[
    "acc_rank", "sign_agree", "rho_adv", "pseudo_contrast",
    "sigma_judge_degenerate", "valid_pair_coverage", "degenerate_share",
    "cos_grad",
]

TiePolicy = Literal["half", "strict", "zero"]


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def advantages(scores: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Group-standardised advantages along the last axis.

    Args:
        scores: ``(..., K)`` reward matrix.
        eps: stabiliser, identical to the training-side value.

    Returns: same shape.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mu = scores.mean(axis=-1, keepdims=True)
    sigma = scores.std(axis=-1, keepdims=True)
    return (scores - mu) / (sigma + eps)


def gold_degrees(scores_gold: np.ndarray) -> np.ndarray:
    """Continuous gold-derived advantage (used only as a diagnostic reference).

    Gold correctness is binary, so this differs from a hard sign only in
    degenerate groups, where it is exactly zero.
    """
    return advantages(scores_gold)


def _pair_index(k: int) -> tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(k, k=1)
    return iu[0], iu[1]


def acc_rank_group(judge: np.ndarray, gold: np.ndarray,
                   tie: TiePolicy = "half") -> float | None:
    """Pairwise ranking accuracy for one group, or ``None`` if undefined."""
    judge = np.asarray(judge, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)
    if gold.std() == 0.0:
        return None
    ii, jj = _pair_index(len(gold))
    dg = gold[ii] - gold[jj]
    keep = dg != 0
    if not keep.any():
        return None
    ii, jj, dg = ii[keep], jj[keep], dg[keep]
    # orient so that "i preferred over j" means gold[i] > gold[j]
    flip = dg < 0
    hi = np.where(flip, jj, ii)
    lo = np.where(flip, ii, jj)
    dj = judge[hi] - judge[lo]
    if tie == "half":
        credit = np.where(dj > 0, 1.0, np.where(dj < 0, 0.0, 0.5))
    elif tie == "strict":
        credit = np.where(dj > 0, 1.0, 0.0)
    elif tie == "zero":
        credit = np.where(dj != 0, (dj > 0).astype(float), 0.0)
    else:  # pragma: no cover
        raise ValueError(tie)
    return float(credit.mean())


def sign_agree_group(adv_j: np.ndarray, adv_g: np.ndarray,
                     deadband: float = 1e-8) -> float | None:
    """Sample-level advantage-sign agreement for one informative group."""
    if np.all(np.asarray(adv_g) == 0):
        return None
    sj = np.sign(np.where(np.abs(adv_j) <= deadband, 0.0, adv_j))
    sg = np.sign(np.where(np.abs(adv_g) <= deadband, 0.0, adv_g))
    return float((sj == sg).mean())


def rho_group(adv_j: np.ndarray, adv_g: np.ndarray) -> float | None:
    """Per-group Pearson correlation of advantages."""
    if np.std(adv_j) == 0 or np.std(adv_g) == 0:
        return None
    return float(np.corrcoef(adv_j, adv_g)[0, 1])


def fisher_z_mean(rs: Iterable[float], ns: Iterable[int]) -> float | None:
    """Weighted mean correlation via Fisher z (n-weighted)."""
    rs = np.asarray(list(rs), dtype=np.float64)
    ns = np.asarray(list(ns), dtype=np.float64)
    if rs.size == 0 or ns.sum() == 0:
        return None
    rs = np.clip(rs, -0.999999, 0.999999)
    z = np.arctanh(rs)
    w = np.maximum(ns - 3.0, 1e-6)
    return float(np.tanh(float((z * w).sum() / w.sum())))


# ---------------------------------------------------------------------------
# per-prompt statistics
# ---------------------------------------------------------------------------


@dataclass
class PerPromptStats:
    """Per-prompt statistic arrays; all share the same prompt ordering."""

    acc_rank: np.ndarray
    sign_agree: np.ndarray
    rho_adv: np.ndarray
    pseudo_contrast: np.ndarray          # bool, degenerate groups only
    sigma_judge_deg: np.ndarray
    informative: np.ndarray              # bool: 0 < #correct < K
    degenerate: np.ndarray               # bool: #correct in {0, K}
    rho_n: np.ndarray                    # K used in the Fisher weighting

    def __len__(self) -> int:
        return len(self.acc_rank)

    def subset(self, idx: np.ndarray) -> "PerPromptStats":
        return PerPromptStats(**{
            f: getattr(self, f)[idx] for f in self.__dataclass_fields__
        })

    @classmethod
    def concat(cls, stats: list["PerPromptStats"]) -> "PerPromptStats":
        """Stack per-prompt arrays from several seeds into one batch."""
        if not stats:
            raise ValueError("concat: empty list")
        fields = cls.__dataclass_fields__
        return cls(**{
            f: np.concatenate([getattr(s, f) for s in stats])
            for f in fields
        })

    @property
    def valid_pair_coverage(self) -> float:
        return float(self.informative.mean()) if len(self) else float("nan")

    @property
    def degenerate_share(self) -> float:
        return float(self.degenerate.mean()) if len(self) else float("nan")


def per_prompt_stats(judge: np.ndarray, gold: np.ndarray, *, tau: float = 0.05,
                     tie: TiePolicy = "half", eps: float = 1e-4) -> PerPromptStats:
    """Compute all fidelity statistics per prompt."""
    judge = np.asarray(judge, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)
    if judge.shape != gold.shape:
        raise ValueError(f"shape mismatch judge={judge.shape} gold={gold.shape}")
    if judge.ndim != 2:
        raise ValueError("expected (P, K) matrices")
    p, k = judge.shape

    acc = np.full(p, np.nan)
    agree = np.full(p, np.nan)
    rho = np.full(p, np.nan)
    pseudo = np.zeros(p, dtype=bool)
    sigma_j_deg = np.full(p, np.nan)
    informative = np.zeros(p, dtype=bool)
    degenerate = np.zeros(p, dtype=bool)

    # gold-degenerate detection is exact, not thresholded
    n_correct = gold.sum(axis=1)
    informative = (n_correct > 0) & (n_correct < k)
    degenerate = ~informative

    adv_j = advantages(judge, eps=eps)
    adv_g = advantages(gold, eps=eps)

    for i in range(p):
        if informative[i]:
            a = acc_rank_group(judge[i], gold[i], tie=tie)
            if a is not None:
                acc[i] = a
                agree[i] = sign_agree_group(adv_j[i], adv_g[i])
            r = rho_group(adv_j[i], adv_g[i])
            if r is not None:
                rho[i] = r
        else:
            spread = float(np.max(judge[i]) - np.min(judge[i]))
            pseudo[i] = spread > tau
            sigma_j_deg[i] = float(np.std(judge[i]))

    return PerPromptStats(
        acc_rank=acc,
        sign_agree=agree,
        rho_adv=rho,
        pseudo_contrast=pseudo,
        sigma_judge_deg=sigma_j_deg,
        informative=informative,
        degenerate=degenerate,
        rho_n=np.full(p, k, dtype=float),
    )


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate(stats: PerPromptStats, metric: MetricName,
              tie_unspecified: float = np.nan) -> float:
    """Aggregate one metric across prompts (nan-aware).

    ``rho_adv`` is aggregated via Fisher z; everything else by nanmean.
    """
    if metric == "acc_rank":
        return float(np.nanmean(stats.acc_rank)) if np.any(~np.isnan(stats.acc_rank)) else tie_unspecified
    if metric == "sign_agree":
        return float(np.nanmean(stats.sign_agree)) if np.any(~np.isnan(stats.sign_agree)) else tie_unspecified
    if metric == "rho_adv":
        mask = ~np.isnan(stats.rho_adv)
        if not mask.any():
            return tie_unspecified
        return fisher_z_mean(stats.rho_adv[mask], stats.rho_n[mask])
    if metric == "pseudo_contrast":
        if not stats.degenerate.any():
            return tie_unspecified
        return float(stats.pseudo_contrast[stats.degenerate].mean())
    if metric == "sigma_judge_degenerate":
        vals = stats.sigma_judge_deg[stats.degenerate]
        vals = vals[~np.isnan(vals)]
        return float(vals.mean()) if vals.size else tie_unspecified
    if metric == "valid_pair_coverage":
        return stats.valid_pair_coverage
    if metric == "degenerate_share":
        return stats.degenerate_share
    raise ValueError(f"unknown metric {metric!r}")


@dataclass
class FidelityReport:
    """All metrics for one (line, snapshot, step) cell."""

    values: dict[str, float]
    n_prompts: int
    n_informative: int
    n_degenerate: int
    tie: TiePolicy = "half"
    tau: float = 0.05
    extra: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float]:
        row = {"n_prompts": self.n_prompts, "n_informative": self.n_informative,
               "n_degenerate": self.n_degenerate}
        row.update(self.values)
        return row


DEFAULT_METRICS: tuple[MetricName, ...] = (
    "acc_rank", "sign_agree", "rho_adv", "pseudo_contrast",
    "sigma_judge_degenerate", "valid_pair_coverage", "degenerate_share",
)


def evaluate_line(judge: np.ndarray, gold: np.ndarray, *,
                  metrics: tuple[MetricName, ...] = DEFAULT_METRICS,
                  tau: float = 0.05, tie: TiePolicy = "half",
                  eps: float = 1e-4) -> FidelityReport:
    """One measurement-line evaluation."""
    stats = per_prompt_stats(judge, gold, tau=tau, tie=tie, eps=eps)
    values = {m: aggregate(stats, m) for m in metrics}
    return FidelityReport(
        values=values,
        n_prompts=len(stats),
        n_informative=int(stats.informative.sum()),
        n_degenerate=int(stats.degenerate.sum()),
        tie=tie,
        tau=tau,
    )


def advantage_mse(adv_j: np.ndarray, adv_g: np.ndarray) -> float:
    """MSE between standardised advantages; equals 2(1-ρ) per group exactly."""
    return float(np.mean((np.asarray(adv_j) - np.asarray(adv_g)) ** 2))


def mse_rho_identity(judge: np.ndarray, gold: np.ndarray, eps: float = 1e-4) -> dict[str, float]:
    """Self-check helper: MSE ≈ 2(1 − ρ) averaged over informative groups.

    Returns observed MSE, the value implied by rho, and their gap.  Used in
    tests and in the run report to prove the identity holds numerically.
    """
    aj, ag = advantages(judge, eps=eps), advantages(gold, eps=eps)
    k = judge.shape[1]
    mses, rhos = [], []
    for i in range(len(judge)):
        if 0 < gold[i].sum() < k:
            mses.append(advantage_mse(aj[i], ag[i]))
            rhos.append(float(np.corrcoef(aj[i], ag[i])[0, 1]))
    if not mses:
        return {"mse": float("nan"), "mse_implied": float("nan"), "gap": float("nan")}
    mse = float(np.mean(mses))
    implied = float(np.mean(2 * (1 - np.asarray(rhos))))
    return {"mse": mse, "mse_implied": implied, "gap": mse - implied}
