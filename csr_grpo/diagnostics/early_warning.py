"""Matched-FPR early warning: Δt_lead and out-of-sample incremental R².

The paper replaces the invalid "one significant, one not" argument with two
pre-registered criteria:

1. **Δt_lead ≥ 50 steps** — under the *same* false-positive rate, calibrated on
   the Continue arm (which defines the null of "no switch"), the step at which
   ``Sign_agree`` (or the degenerate pseudo-contrast rate) first flags
   degradation is earlier than the step at which ``Acc_rank`` does.
2. **ΔR²_oos > 0** — adding ``Sign_agree`` to a cross-step regression that
   predicts the future forgetting measure has out-of-sample predictive power
   beyond ``Acc_rank`` alone.

Both are computed here; neither decides causality (E2 does).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

Direction = Literal["decrease", "increase"]


# ---------------------------------------------------------------------------
# threshold calibration on the control arm
# ---------------------------------------------------------------------------


@dataclass
class Detector:
    """A one-sided threshold calibrated to a target false-positive rate."""

    threshold: float
    target_fpr: float
    direction: Direction
    control_values: np.ndarray
    achieved_fpr: float

    def triggered(self, value: float) -> bool:
        if np.isnan(value):
            return False
        return value <= self.threshold if self.direction == "decrease" else value >= self.threshold


def calibrate(control_series: np.ndarray, target_fpr: float = 0.05,
              direction: Direction = "decrease") -> Detector:
    """Pick the threshold whose empirical FPR on the control arm equals ``target_fpr``.

    Using the empirical quantile is deliberate: it makes the two metrics
    comparable even when their sampling distributions differ, which is exactly
    the "matched FPR" requirement.
    """
    vals = np.asarray(control_series, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        raise ValueError("control arm is empty")
    if not (0 < target_fpr < 1):
        raise ValueError("target_fpr must be in (0,1)")
    q = target_fpr if direction == "decrease" else 1 - target_fpr
    thr = float(np.quantile(vals, q))
    achieved = float(np.mean(vals <= thr) if direction == "decrease"
                     else np.mean(vals >= thr))
    return Detector(threshold=thr, target_fpr=target_fpr, direction=direction,
                    control_values=vals, achieved_fpr=achieved)


def first_detection(series: np.ndarray, detector: Detector,
                    steps: Sequence[int] | None = None,
                    consecutive: int = 1) -> int | None:
    """First step index (or actual step label) meeting the trigger."""
    vals = np.asarray(series, dtype=float)
    run = 0
    for i, v in enumerate(vals):
        if detector.triggered(v):
            run += 1
            if run >= consecutive:
                return int(steps[i]) if steps is not None else i
        else:
            run = 0
    return None


# ---------------------------------------------------------------------------
# Δt_lead
# ---------------------------------------------------------------------------


@dataclass
class LeadTimeResult:
    per_seed: dict[str, list[int | None]]
    delta_per_seed: list[float]
    mean_delta: float
    lo: float
    hi: float
    criterion_met: bool
    target_fpr: float
    min_required: int
    # False when the control arm has too few points to resolve the requested
    # false-positive rate: a 5% threshold cannot be estimated from 6 samples,
    # and reporting "no lead" there would be a false negative in disguise.
    resolvable: bool = True
    n_control_points: int = 0

    @property
    def status(self) -> str:
        if not self.resolvable:
            return "insufficient_resolution"
        if not self.delta_per_seed:
            return "no_detection"
        return "measured"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "resolvable": self.resolvable,
            "n_control_points": self.n_control_points,
            "delta_t_lead_mean": self.mean_delta,
            "ci": [self.lo, self.hi],
            "criterion_met": self.criterion_met,
            "target_fpr": self.target_fpr,
            "min_required": self.min_required,
            "per_seed": {k: list(v) for k, v in self.per_seed.items()},
            "delta_per_seed": self.delta_per_seed,
        }


def matched_fpr_lead_time(
    switch_curves: dict[str, list[np.ndarray]],
    control_curves: dict[str, list[np.ndarray]],
    steps: Sequence[int],
    *,
    early_metric: str = "sign_agree",
    late_metric: str = "acc_rank",
    target_fpr: float = 0.05,
    min_required: int = 50,
    direction: Direction = "decrease",
    n_resamples: int = 2000,
    seed: int = 0,
) -> LeadTimeResult:
    """Compute per-seed ``Δt_lead`` with a bootstrap CI over seeds.

    Args:
        switch_curves: metric -> list over seeds of the Switch arm series.
        control_curves: same shape for the Continue arm (defines the null).
        steps: step labels aligned with each series.
    """
    n_seeds = len(switch_curves[early_metric])
    if n_seeds != len(switch_curves[late_metric]):
        raise ValueError("metric series disagree on seed count")

    per_seed: dict[str, list[int | None]] = {early_metric: [], late_metric: []}
    deltas: list[float] = []
    for s in range(n_seeds):
        det_e = calibrate(control_curves[early_metric][s], target_fpr, direction)
        det_l = calibrate(control_curves[late_metric][s], target_fpr, direction)
        t_e = first_detection(switch_curves[early_metric][s], det_e, steps, 2)
        t_l = first_detection(switch_curves[late_metric][s], det_l, steps, 2)
        per_seed[early_metric].append(t_e)
        per_seed[late_metric].append(t_l)
        if t_e is not None and t_l is not None:
            deltas.append(float(t_l - t_e))
        elif t_l is not None and t_e is None:
            # early metric never fired within the window: treat as no lead
            deltas.append(0.0)

    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        return LeadTimeResult(per_seed, [], float("nan"), float("nan"),
                              float("nan"), False, target_fpr, min_required)
    # Can the control arm even resolve the requested FPR?  Estimating a 5%
    # quantile from 6 points picks the sample minimum, so the "threshold" is
    # pure noise.  Require at least one expected exceedance.
    n_ctrl = min(len(c) for c in control_curves[early_metric])
    resolvable = (n_ctrl * target_fpr) >= 1.0

    if arr.size == 0:
        return LeadTimeResult(per_seed, [], float("nan"), float("nan"),
                              float("nan"), False, target_fpr, min_required,
                              resolvable=resolvable, n_control_points=n_ctrl)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(arr, size=arr.size, replace=True).mean()
                     for _ in range(n_resamples)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    mean = float(arr.mean())
    return LeadTimeResult(
        per_seed=per_seed,
        delta_per_seed=[float(x) for x in arr],
        mean_delta=mean,
        lo=float(lo),
        hi=float(hi),
        criterion_met=bool(resolvable and mean >= min_required and lo > 0),
        target_fpr=target_fpr,
        min_required=min_required,
        resolvable=resolvable,
        n_control_points=n_ctrl,
    )


# ---------------------------------------------------------------------------
# ΔR²_oos
# ---------------------------------------------------------------------------


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def _ols_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return X1 @ beta


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


@dataclass
class IncrementalR2:
    r2_base: float
    r2_full: float
    delta_oos: float
    ci: tuple[float, float]
    n_folds: int
    criterion_met: bool
    min_delta: float = 0.0

    def as_dict(self) -> dict:
        return {
            "r2_acc_rank_only": self.r2_base,
            "r2_with_sign_agree": self.r2_full,
            "delta_r2_oos": self.delta_oos,
            "ci": list(self.ci),
            "n_folds": self.n_folds,
            "criterion_met": self.criterion_met,
        }


def predictive_gain(
    features_base: np.ndarray,
    features_extra: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    min_delta: float = 0.0,
) -> IncrementalR2:
    """Leave-one-group-out OOS R² with and without the extra metric.

    ``groups`` is typically the seed index (or a blocked time index) so that
    folds respect dependence between consecutive steps.
    """
    Xb = np.asarray(features_base, dtype=float)
    Xe = np.asarray(features_extra, dtype=float)
    Xfull = np.column_stack([Xb, Xe])
    y = np.asarray(target, dtype=float)
    g = np.asarray(groups)

    uniq = np.unique(g)
    yhat_b = np.empty_like(y)
    yhat_f = np.empty_like(y)
    for u in uniq:
        te = g == u
        if te.all():
            raise ValueError("a single group holds every row; cannot hold out")
        bb = _ols_fit(Xb[~te], y[~te])
        bf = _ols_fit(Xfull[~te], y[~te])
        yhat_b[te] = _ols_predict(Xb[te], bb)
        yhat_f[te] = _ols_predict(Xfull[te], bf)

    r2b, r2f = _r2(y, yhat_b), _r2(y, yhat_f)

    rng = np.random.default_rng(seed)
    gains = np.empty(n_resamples)
    n = len(y)
    for r in range(n_resamples):
        # draw WITH replacement — do NOT take np.unique; duplicates are the
        # bootstrap mechanism and dropping them biases the variance estimate
        idx = rng.integers(0, n, size=n)
        gains[r] = _r2(y[idx], yhat_f[idx]) - _r2(y[idx], yhat_b[idx])
    lo, hi = np.quantile(gains, [0.025, 0.975])
    delta = r2f - r2b
    return IncrementalR2(
        r2_base=float(r2b), r2_full=float(r2f), delta_oos=float(delta),
        ci=(float(lo), float(hi)), n_folds=int(len(uniq)),
        criterion_met=bool(delta > min_delta and lo > 0),
    )


# ---------------------------------------------------------------------------
# combined structural-lag verdict (H1's second half)
# ---------------------------------------------------------------------------


@dataclass
class LagVerdict:
    lead: LeadTimeResult
    r2: IncrementalR2 | None = None

    @property
    def met(self) -> bool:
        """Criterion 1 OR criterion 2 (as pre-registered)."""
        return self.lead.criterion_met or bool(self.r2 and self.r2.criterion_met)

    def as_dict(self) -> dict:
        return {
            "dt_lead": self.lead.as_dict(),
            "delta_r2_oos": self.r2.as_dict() if self.r2 else None,
            "structural_lag_supported": self.met,
        }
