"""Prompt-level (paired) cluster bootstrap, non-inferiority test, effect triage.

Why clusters: pairs inside a prompt share the prompt's difficulty and topic, so
they are correlated (ICC = ρ_intra).  Resampling *pairs* understates variance;
we resample prompt IDs instead and recompute the statistic on the resample
(Parametric Cluster-Bootstrap, §"Component 2"-3 of the plan).

Why paired: every cell of a 2x2 table or every arm of E1/E2 is measured on the
same prompts.  One resample of prompt IDs is drawn, and *every* cell is
recomputed on that same resample, so differences are within-prompt and the
correlation structure is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

from .fidelity import MetricName, PerPromptStats, aggregate

Decision = Literal["go", "inconclusive", "equivalence"]


# ---------------------------------------------------------------------------
# paired importance resampling
# ---------------------------------------------------------------------------


def bootstrap_indices(n: int, n_resamples: int, rng: np.random.Generator,
                      strata: np.ndarray | None = None) -> np.ndarray:
    """Return ``(n_resamples, n)`` index matrix of prompt resamples.

    ``strata`` optionally reweights so each bootstrap resample preserves the
    empirical distribution of a stratum (e.g. task domain).
    """
    if strata is None:
        return rng.integers(0, n, size=(n_resamples, n))
    strata = np.asarray(strata)
    out = np.empty((n_resamples, n), dtype=np.int64)
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        out[:, members] = members[draws]
    return out


def _stat_on(stats: PerPromptStats, metric: MetricName,
             idx: np.ndarray) -> float:
    return aggregate(stats.subset(idx), metric)


def cluster_statistic(stats: PerPromptStats, metric: MetricName,
                      idx_boot: np.ndarray) -> np.ndarray:
    """Vectorised statistic over bootstrap resamples."""
    return np.array([_stat_on(stats, metric, idx) for idx in idx_boot])


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    boot: np.ndarray
    level: float = 0.95
    # resamples that were undefined (e.g. only degenerate groups) and dropped
    n_dropped: int = 0
    # A CI is only reported as a decision input when it is both finite and
    # computed on (almost) the whole bootstrap distribution.  Dropping
    # undefined resamples is not neutral: the ones that vanish are exactly
    # "no informative prompt was drawn", so the survivor distribution is
    # conditional and its quantiles are biased.  At P_audit = 200-400 measured
    # degeneracy this never bites; at a toy-sized P with 85% degenerate groups
    # 2.8% of resamples go missing.
    max_drop_fraction: float = 0.05

    @property
    def drop_fraction(self) -> float:
        total = self.n_dropped + int(np.isfinite(self.boot).sum())
        return self.n_dropped / total if total else float("nan")

    @property
    def usable(self) -> bool:
        """Finite *and* computed on essentially the whole distribution."""
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            return False
        frac = self.drop_fraction
        return bool(np.isfinite(frac) and frac <= self.max_drop_fraction)

    def excludes(self, value: float = 0.0) -> bool:
        return (self.lo > value) or (self.hi < value)

    def within(self, halfwidth: float) -> bool:
        return self.lo >= -halfwidth and self.hi <= halfwidth

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi,
                "level": self.level, "n_dropped_resamples": self.n_dropped,
                "drop_fraction": self.drop_fraction, "usable": self.usable}


def percentile_ci(samples: np.ndarray, point: float, level: float = 0.95) -> BootstrapCI:
    """Percentile CI, with NaN resamples dropped rather than propagated.

    A resample can be undefined (e.g. it happened to draw only degenerate
    groups, where ``Acc_rank`` has no denominator).  ``np.quantile`` would return
    ``nan`` for the entire interval in that case, silently destroying a valid
    result.  We drop the undefined resamples and record how many were lost, so a
    heavily-degenerate probe is visible instead of invisible.
    """
    arr = np.asarray(samples, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    dropped = int(arr.size - finite.size)
    if finite.size == 0:
        return BootstrapCI(point=float(point), lo=float("nan"), hi=float("nan"),
                           boot=arr, level=level, n_dropped=dropped)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(finite, [alpha, 1 - alpha])
    return BootstrapCI(point=float(point), lo=float(lo), hi=float(hi),
                       boot=arr, level=level, n_dropped=dropped)


# ---------------------------------------------------------------------------
# paired multi-cell bootstrap
# ---------------------------------------------------------------------------


@dataclass
class PairedBootstrapResult:
    cells: dict[str, BootstrapCI]
    contrasts: dict[str, BootstrapCI]
    rng_seed: int
    n_resamples: int
    n_prompts: int

    def as_dict(self) -> dict:
        return {
            "cells": {k: v.as_dict() for k, v in self.cells.items()},
            "contrasts": {k: v.as_dict() for k, v in self.contrasts.items()},
            "n_resamples": self.n_resamples,
            "n_prompts": self.n_prompts,
        }


def paired_cluster_bootstrap(
    cells: Mapping[str, PerPromptStats],
    metric: MetricName,
    contrasts: Sequence[tuple[str, str]] = (),
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    strata: np.ndarray | None = None,
) -> PairedBootstrapResult:
    """Bootstrap several cells sharing one prompt index, plus their contrasts.

    Args:
        cells: name -> ``PerPromptStats``; all must share prompt ordering.
        contrasts: list of ``(a, b)`` meaning ``a - b``.
    """
    names = list(cells)
    n = len(cells[names[0]])
    for name, st in cells.items():
        if len(st) != n:
            raise ValueError(f"cell {name} has {len(st)} prompts, expected {n}")

    rng = np.random.default_rng(seed)
    idx_boot = bootstrap_indices(n, n_resamples, rng, strata=strata)

    boot: dict[str, np.ndarray] = {}
    point: dict[str, float] = {}
    for name, st in cells.items():
        point[name] = aggregate(st, metric)
        boot[name] = cluster_statistic(st, metric, idx_boot)

    cell_ci = {name: percentile_ci(boot[name], point[name], level) for name in names}

    contrast_ci: dict[str, BootstrapCI] = {}
    for a, b in contrasts:
        if a not in cells or b not in cells:
            raise KeyError(f"contrast {a}-{b} references unknown cell")
        key = f"{a}-{b}"
        samples = boot[a] - boot[b]
        contrast_ci[key] = percentile_ci(samples, point[a] - point[b], level)

    return PairedBootstrapResult(cells=cell_ci, contrasts=contrast_ci,
                                 rng_seed=seed, n_resamples=n_resamples,
                                 n_prompts=n)


# ---------------------------------------------------------------------------
# one-sided non-inferiority (H6)
# ---------------------------------------------------------------------------


@dataclass
class NonInferiorityResult:
    delta: float                 # observed after - before
    margin: float                # δ_NI (positive, in same units as metric)
    lo: float
    hi: float
    z: float
    p_value: float               # H0: delta <= -margin
    non_inferior: bool
    level: float
    two_sided_p: float           # TOST p for the secondary report

    def as_dict(self) -> dict:
        return {
            "delta": self.delta, "margin": self.margin,
            "ci": [self.lo, self.hi], "z": self.z,
            "p_one_sided": self.p_value, "p_tost": self.two_sided_p,
            "non_inferior": self.non_inferior, "level": self.level,
        }


def non_inferiority_test(paired_diff: np.ndarray, margin: float, *,
                         level: float = 0.95,
                         cluster_ids: np.ndarray | None = None) -> NonInferiorityResult:
    """One-sided paired test of ``H0: Δ ≤ −δ_NI`` at prompt-cluster level.

    Args:
        paired_diff: per-prompt difference ``after − before``.
        margin: δ_NI, positive.
    """
    from scipy import stats

    d = np.asarray(paired_diff, dtype=float)
    d = d[~np.isnan(d)]
    n = d.size
    if n < 2:
        raise ValueError("need at least 2 prompts")
    mean = float(d.mean())
    if cluster_ids is not None:
        # cluster-robust SE (Liang-Zeger on the mean)
        cid = np.asarray(cluster_ids)
        uniq = np.unique(cid)
        sums = np.array([d[cid == u].sum() for u in uniq])
        counts = np.array([(cid == u).sum() for u in uniq])
        var = float((counts ** 2 * (sums / counts - mean) ** 2).sum()) / (n ** 2)
        se = np.sqrt(var)
    else:
        se = float(d.std(ddof=1) / np.sqrt(n))
    se = max(se, 1e-12)
    z_stat = (mean + margin) / se
    p_one = float(stats.norm.sf(z_stat))          # reject H0 for small p
    zc = float(stats.norm.ppf(level))
    lo, hi = mean - zc * se, mean + zc * se
    # TOST: equivalence within ±margin needs both one-sided tests to reject
    p_low = float(stats.norm.sf((mean + margin) / se))
    p_high = float(stats.norm.cdf((mean - margin) / se))
    return NonInferiorityResult(
        delta=mean, margin=margin, lo=float(lo), hi=float(hi), z=float(z_stat),
        p_value=p_one, non_inferior=bool(lo > -margin),
        level=level, two_sided_p=float(max(p_low, p_high)),
    )


# ---------------------------------------------------------------------------
# effect triage: resource stop vs scientific equivalence
# ---------------------------------------------------------------------------

SESOI_UNITS = "fraction"  # metrics are fractions in [0,1]; SESOI given in points


def classify_effect(ci: BootstrapCI, *, sesoi_pt: float = 2.0,
                    direction: Literal["greater", "less"] = "greater") -> Decision:
    """Three-way pre-registered decision (paper: Go / equivalence / inconclusive).

    * ``go``            – CI excludes 0 in the hypothesised direction
    * ``equivalence``   – CI lies entirely inside [-sesoi, +sesoi]: a *scientific*
                          conclusion that no meaningful effect exists
    * ``inconclusive``  – neither: CI contains 0 but is wider than the band,
                          i.e. underpowered
    """
    # An undefined or heavily-truncated bootstrap distribution cannot support
    # *any* of the three verdicts: "go" and "equivalence" are both claims about
    # the sampling distribution, and half of it is missing.  Reporting
    # "inconclusive" is the honest answer, and it is the same verdict the
    # pre-registration assigns to an underpowered run.
    if not ci.usable:
        return "inconclusive"
    band = sesoi_pt / 100.0
    inside = ci.lo >= -band and ci.hi <= band
    if inside:
        return "equivalence"
    if direction == "greater" and ci.lo > 0:
        return "go"
    if direction == "less" and ci.hi < 0:
        return "go"
    return "inconclusive"


def required_prompts(p_d: float, icc: float, pairs_per_prompt: int = 4,
                     delta_pt: float = 2.5, alpha: float = 0.05,
                     power: float = 0.80) -> int:
    """Step-0 power simulation helper (analytic approximation).

    Two-stage: (1) pairs needed for a paired non-inferiority test where
    discordant-pair rate is ``p_d``; (2) inflate by the design effect
    ``DEFF = 1 + (m-1)ρ_intra`` to convert pairs to prompts.
    """
    from scipy import stats

    if not (0 < p_d < 1):
        raise ValueError("p_d must be in (0, 1)")
    delta = delta_pt / 100.0
    # paired difference between two Bernoulli accuracy indicators has variance
    # p_d - (p_d)^2 scaled: conservative var = p_d * (1 - p_d) / n_pairs
    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    n_pairs = ((z_a + z_b) ** 2) * p_d * (1 - p_d) / (delta ** 2)
    deff = 1 + (pairs_per_prompt - 1) * max(icc, 0.0)
    return int(np.ceil(n_pairs * deff / pairs_per_prompt))


def simulate_power(p_d: float, icc: float, n_prompts: int,
                   pairs_per_prompt: int = 4, delta_pt: float = 2.5,
                   alpha: float = 0.05, n_sims: int = 500,
                   seed: int = 0) -> float:
    """Monte-Carlo power for the prompt-clustered paired NI test."""
    from scipy import stats

    rng = np.random.default_rng(seed)
    delta = delta_pt / 100.0
    # prompt-level random effect with total variance consistent with icc
    sd_prompt = np.sqrt(icc) * np.sqrt(p_d * (1 - p_d))
    sd_within = np.sqrt(max(p_d * (1 - p_d) - sd_prompt ** 2, 1e-9))
    hits = 0
    for _ in range(n_sims):
        eff = rng.normal(0.0, sd_prompt, size=n_prompts)
        diffs = eff[:, None] + rng.normal(0.0, sd_within / np.sqrt(pairs_per_prompt),
                                          size=(n_prompts, pairs_per_prompt))
        per_prompt = diffs.mean(axis=1)
        # H0: delta <= -margin ; reject when lower CI bound > -margin
        n = per_prompt.size
        se = per_prompt.std(ddof=1) / np.sqrt(n)
        lo = per_prompt.mean() - stats.norm.ppf(0.95) * se
        hits += int(lo > -delta)
    return hits / n_sims
