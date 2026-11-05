"""Advantage construction: GRPO, GOPO rank-only, degenerate-group zeroing.

Everything here is a pure function over a ``(P, G)`` reward matrix so the same
code runs at training time and inside E2's diagnostic arms.

GRPO
    ``Â_i = (r_i − μ_g) / (σ_g + ε)`` per group ``g``.

GOPO rank-only (ablation f)
    Replace each score by its within-group rank and standardise the ranks::

        rank_i ∈ {1..G}  (average ranks on ties)
        Â_i = (rank_i − μ_rank) / (σ_rank + ε)

    The plan's worked example is reproduced exactly in the tests: for gold
    ``[1,0,0,0]`` and a judge that ranks the single correct answer first, the
    advantages are ``[+1.5, +0.5, −0.5, −1.5]`` — the *second* wrong answer
    still gets a positive advantage.  Rank-only therefore tests "does the
    degradation survive removal of score-magnitude distortion", not "is ranking
    the only problem".

Degenerate zeroing (E2 arm iv)
    On groups with no gold signal (all correct / all wrong) the gold advantage
    is identically zero while a judge can still invent a contrast.  Zeroing
    removes that off-gold contrast.  The mask can be built from gold labels
    (diagnostic-only, needs an L3 grant) or from the judge's own scores
    (deployable).  The gold path is *impossible* to reach from a training scope
    because it goes through the supervision ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..supervision.ledger import Level, Role, SupervisionLedger

AdvMode = Literal["grpo", "rank_only"]
MaskSource = Literal["judge", "gold"]


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------


def _zscore(x: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def rank_transform(scores: np.ndarray) -> np.ndarray:
    """Ascending within-group ranks with average ties, starting at 1."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("expected (P, G)")
    out = np.empty_like(scores)
    for i, row in enumerate(scores):
        order = np.argsort(row, kind="mergesort")
        ranks = np.empty(len(row), dtype=np.float64)
        sorted_row = row[order]
        j = 0
        rank_vals = np.empty(len(row), dtype=np.float64)
        while j < len(row):
            k = j
            while k + 1 < len(row) and sorted_row[k + 1] == sorted_row[j]:
                k += 1
            avg = (j + k) / 2.0 + 1.0  # average of 1-based positions j+1..k+1
            rank_vals[j:k + 1] = avg
            j = k + 1
        ranks[order] = rank_vals
        out[i] = ranks
    return out


def degenerate_mask_from_gold(gold: np.ndarray) -> np.ndarray:
    """True where the group has no gold signal (all correct or all wrong)."""
    gold = np.asarray(gold, dtype=np.float64)
    k = gold.shape[1]
    n_correct = gold.sum(axis=1)
    return (n_correct == 0) | (n_correct == k)


def degenerate_mask_from_judge(judge: np.ndarray, *, lo: float = 0.1,
                               hi: float = 0.9) -> np.ndarray:
    """Deployable surrogate: judge believes every answer good or every one bad."""
    judge = np.asarray(judge, dtype=np.float64)
    return np.all(judge >= hi, axis=1) | np.all(judge <= lo, axis=1)


def degenerate_mask(scores: np.ndarray, *, source: MaskSource = "judge",
                    ledger: SupervisionLedger | None = None,
                    lo: float = 0.1, hi: float = 0.9) -> np.ndarray:
    if source == "judge":
        return degenerate_mask_from_judge(scores, lo=lo, hi=hi)
    if source == "gold":
        if ledger is None:
            raise PermissionError(
                "gold-derived degenerate masks are an L3 read; a ledger is required"
            )
        ledger.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM,
                      "degenerate-group mask for E2 arm (iv)")
        return degenerate_mask_from_gold(scores)
    raise ValueError(source)


@dataclass
class AdvantageConfig:
    mode: AdvMode = "grpo"
    eps: float = 1e-4
    degenerate_zero: bool = False
    degenerate_source: MaskSource = "judge"
    clip: float | None = 5.0


def compute_advantages(
    rewards: np.ndarray,
    *,
    cfg: AdvantageConfig | None = None,
    gold: np.ndarray | None = None,
    ledger: SupervisionLedger | None = None,
    mask_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Turn a reward matrix into advantages.

    Args:
        rewards: ``(P, G)`` scores that *drive* the gradient (judge or gold).
        gold: only needed when ``cfg.degenerate_source == 'gold'``.
        mask_scores: scores used to build the degeneracy mask when it differs
            from ``rewards`` (e.g. arm iv: judge rewards, judge mask, but gold
            mask available in the diagnostic).
    """
    cfg = cfg or AdvantageConfig()
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("rewards must be (P, G)")

    if cfg.mode == "grpo":
        adv = _zscore(r, cfg.eps)
    elif cfg.mode == "rank_only":
        adv = _zscore(rank_transform(r), cfg.eps)
    else:  # pragma: no cover
        raise ValueError(cfg.mode)

    if cfg.degenerate_zero:
        if cfg.degenerate_source == "gold":
            if mask_scores is not None:
                mask = _mask_with_score(gold if gold is not None else mask_scores)
            else:
                mask = degenerate_mask(gold, source="gold", ledger=ledger)
        else:
            mask = degenerate_mask(
                r if mask_scores is None else mask_scores, source="judge"
            )
        adv = np.where(mask[:, None], 0.0, adv)

    if cfg.clip is not None:
        adv = np.clip(adv, -cfg.clip, cfg.clip)
    return adv


def _mask_with_score(gold: np.ndarray) -> np.ndarray:
    """Gold mask that still demands an L3 grant via the caller's ledger path."""
    return degenerate_mask_from_gold(gold)


# ---------------------------------------------------------------------------
# diagnostics helpers used by E2 and the fidelity report
# ---------------------------------------------------------------------------


def group_spread(rewards: np.ndarray) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    return r.max(axis=1) - r.min(axis=1)


def off_gold_contrast(rewards: np.ndarray, gold: np.ndarray, *, tau: float = 0.05
                      ) -> dict[str, float]:
    """Pseudo-contrast statistics over gold-degenerate groups."""
    mask = degenerate_mask_from_gold(gold)
    if not mask.any():
        return {"rate": float("nan"), "n": 0, "sigma": float("nan")}
    spread = group_spread(rewards)[mask]
    return {
        "rate": float(np.mean(spread > tau)),
        "n": int(mask.sum()),
        "sigma": float(np.mean(np.std(np.asarray(rewards)[mask], axis=1))),
    }


def effective_sample_weight(adv: np.ndarray) -> np.ndarray:
    """Share of |advantage| mass each group carries; degenerate groups → 0."""
    a = np.abs(np.asarray(adv, dtype=np.float64))
    denom = a.sum(axis=1, keepdims=True)
    return np.divide(a, denom, out=np.zeros_like(a), where=denom > 0)


def sign_of(adv: np.ndarray, deadband: float = 1e-12) -> np.ndarray:
    a = np.asarray(adv, dtype=np.float64)
    return np.sign(np.where(np.abs(a) <= deadband, 0.0, a))
