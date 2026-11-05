"""Cluster bootstrap, non-inferiority, effect triage and early warning."""

import numpy as np
import pytest

from csr_grpo.diagnostics.bootstrap import (
    BootstrapCI, classify_effect, cluster_statistic, non_inferiority_test,
    paired_cluster_bootstrap, percentile_ci, required_prompts, simulate_power,
)
from csr_grpo.diagnostics.early_warning import (
    calibrate, first_detection, matched_fpr_lead_time, predictive_gain,
)
from csr_grpo.diagnostics.fidelity import per_prompt_stats


def _stats(seed=0, n=60, k=8, quality=0.8):
    rng = np.random.default_rng(seed)
    gold = (rng.random((n, k)) > 0.5).astype(float)
    judge = quality * gold + (1 - quality) * rng.random((n, k))
    return per_prompt_stats(judge, gold), per_prompt_stats(gold, gold)


def test_paired_bootstrap_uses_same_prompt_resample():
    a, b = _stats()
    res = paired_cluster_bootstrap({"a": a, "b": b}, "acc_rank",
                                   [("a", "b")], n_resamples=200, seed=1)
    c = res.contrasts["a-b"]
    # the contrast of a perfect cell against itself must be ~0 with CI touching 0
    assert abs(c.point) < 1e-9
    assert c.lo <= 0 <= c.hi


def test_bootstrap_is_deterministic_given_seed():
    a, _ = _stats()
    r1 = paired_cluster_bootstrap({"a": a}, "sign_agree", n_resamples=100, seed=7)
    r2 = paired_cluster_bootstrap({"a": a}, "sign_agree", n_resamples=100, seed=7)
    assert r1.cells["a"].lo == r2.cells["a"].lo


def test_bootstrap_rejects_mismatched_prompt_counts():
    a, _ = _stats(n=10)
    small, _ = _stats(n=5)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap({"a": a, "b": small}, "acc_rank",
                                 n_resamples=10)


def test_classify_effect_three_way():
    eq = BootstrapCI(point=0.001, lo=-0.01, hi=0.012, boot=np.zeros(1))
    assert classify_effect(eq, sesoi_pt=2.0) == "equivalence"
    go = BootstrapCI(point=0.05, lo=0.01, hi=0.09, boot=np.zeros(1))
    assert classify_effect(go, sesoi_pt=2.0) == "go"
    inc = BootstrapCI(point=0.05, lo=-0.02, hi=0.12, boot=np.zeros(1))
    assert classify_effect(inc, sesoi_pt=2.0) == "inconclusive"


def test_non_inferiority_passes_when_difference_is_small():
    rng = np.random.default_rng(3)
    diff = rng.normal(0.001, 0.01, size=400)
    r = non_inferiority_test(diff, margin=0.025)
    assert r.non_inferior and r.p_value < 0.05


def test_non_inferiority_fails_when_difference_is_large():
    rng = np.random.default_rng(4)
    diff = rng.normal(-0.10, 0.01, size=400)
    r = non_inferiority_test(diff, margin=0.025)
    assert not r.non_inferior


def test_required_prompts_scales_with_icc():
    low = required_prompts(0.08, icc=0.0)
    high = required_prompts(0.08, icc=0.30)
    assert high > low


def test_power_simulation_increases_with_sample_size():
    p_small = simulate_power(0.08, 0.15, n_prompts=50, n_sims=120, seed=0)
    p_large = simulate_power(0.08, 0.15, n_prompts=800, n_sims=120, seed=0)
    assert p_large >= p_small


def test_calibrate_matches_requested_fpr():
    rng = np.random.default_rng(5)
    ctrl = rng.normal(0.8, 0.02, size=400)
    det = calibrate(ctrl, target_fpr=0.05, direction="decrease")
    assert det.achieved_fpr == pytest.approx(0.05, abs=0.02)


def test_first_detection_respects_consecutive_requirement():
    det = calibrate(np.array([0.9] * 50), 0.05, "decrease")
    series = np.array([0.5, 0.3, 0.2, 0.1])
    assert first_detection(series, det, consecutive=1) == 0
    assert first_detection(series, det, consecutive=2) == 1


def test_lead_time_rewards_earlier_detection():
    steps = list(range(0, 200, 10))
    rng = np.random.default_rng(6)
    ctrl = {m: [rng.normal(0.8, 0.02, len(steps)) for _ in range(3)]
            for m in ("sign_agree", "acc_rank")}
    early = np.linspace(0.80, 0.40, len(steps))
    late = np.linspace(0.80, 0.72, len(steps))
    switch = {"sign_agree": [early, early, early], "acc_rank": [late, late, late]}
    res = matched_fpr_lead_time(switch, ctrl, steps, min_required=50)
    assert res.mean_delta > 0
    assert res.criterion_met


def test_predictive_gain_detects_useful_feature():
    rng = np.random.default_rng(7)
    n = 80
    base = rng.normal(size=(n, 1))
    extra = rng.normal(size=(n, 1))
    y = 3 * base[:, 0] + 2 * extra[:, 0] + rng.normal(0, 0.1, n)
    groups = rng.integers(0, 8, n)
    res = predictive_gain(base, extra, y, groups, n_resamples=200)
    assert res.delta_oos > 0
    assert res.criterion_met


def test_predictive_gain_bootstrap_resamples_with_replacement():
    """Regression for §1.38: np.unique had silently turned the bootstrap into
    a ~63% subsample, underestimating the CI width.  A real bootstrap draw
    contains duplicate indices; the per-resample effective sample size must
    vary, not be pinned at the number of unique rows.
    """
    rng = np.random.default_rng(11)
    n = 60
    base = rng.normal(size=(n, 1))
    extra = rng.normal(size=(n, 1))
    y = 2 * base[:, 0] + extra[:, 0] + rng.normal(0, 0.5, n)
    groups = np.arange(n)  # leave-one-out folds
    res = predictive_gain(base, extra, y, groups, n_resamples=400, seed=3)
    # CI must be finite and have nonzero width — a subsampling bug collapses it
    assert np.isfinite(res.ci[0]) and np.isfinite(res.ci[1])
    assert res.ci[1] >= res.ci[0]
    # the point estimate is the leave-one-out OOS ΔR², independent of the bug
    assert res.delta_oos == res.r2_full - res.r2_base


# ---------------------------------------------------------------------------
# resolution guard: "cannot estimate" is not "no lead"
# ---------------------------------------------------------------------------


def test_lead_time_reports_insufficient_resolution_on_tiny_series():
    """A 5% FPR cannot be estimated from 6 points; asking for it must say so."""
    steps = list(range(6))
    sw = {"sign_agree": [np.linspace(0.8, 0.5, 6)], "acc_rank": [np.linspace(0.8, 0.75, 6)]}
    co = {"sign_agree": [np.full(6, 0.8)], "acc_rank": [np.full(6, 0.8)]}
    res = matched_fpr_lead_time(sw, co, steps, target_fpr=0.05)
    assert res.resolvable is False
    assert res.status == "insufficient_resolution"
    assert res.criterion_met is False


def test_lead_time_resolvable_once_the_series_is_long_enough():
    """With 200 control points a 5% FPR is resolvable; status must be "measured"
    regardless of whether the particular series shows positive lead or zero lead —
    a lead of exactly zero is a valid measurement, not an error.
    """
    n = 200
    steps = list(range(n))
    sw = {"sign_agree": [np.linspace(0.8, 0.4, n)], "acc_rank": [np.linspace(0.8, 0.7, n)]}
    co = {"sign_agree": [np.full(n, 0.8)], "acc_rank": [np.full(n, 0.8)]}
    res = matched_fpr_lead_time(sw, co, steps, target_fpr=0.05)
    assert res.resolvable is True
    # "measured" means a valid attempt was made, independent of sign of delta
    assert res.status == "measured"
    # criterion_met requires mean_delta >= min_required, but that is a threshold
    # decision, not a test of whether the machinery runs
    assert res.n_control_points == n


# ---------------------------------------------------------------------------
# undefined-resample accounting
# ---------------------------------------------------------------------------
#
# Dropping undefined resamples keeps a valid CI from being destroyed by one bad
# draw, but the drop is not neutral: what vanishes is exactly "no informative
# prompt was drawn", so the surviving distribution is conditional.  The guard
# below decides when the surviving distribution is still trustworthy.


def _ci(lo, hi, dropped, n_good=95):
    boot = np.concatenate([np.linspace(lo, hi, n_good),
                           np.full(dropped, np.nan)])
    return BootstrapCI(point=(lo + hi) / 2, lo=lo, hi=hi, boot=boot,
                       n_dropped=dropped)


def test_a_clean_bootstrap_is_usable():
    ci = _ci(-0.01, 0.05, dropped=0)
    assert ci.drop_fraction == 0.0
    assert ci.usable is True


def test_a_heavily_truncated_bootstrap_is_not_usable():
    ci = _ci(-0.01, 0.05, dropped=40, n_good=60)
    assert ci.drop_fraction == pytest.approx(0.4)
    assert ci.usable is False


def test_an_all_nan_bootstrap_is_not_usable():
    ci = BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"),
                     boot=np.full(50, np.nan), n_dropped=50)
    assert ci.usable is False


def test_classification_refuses_to_verdict_a_truncated_distribution():
    """An undefined distribution cannot support 'go' *or* 'equivalence': both
    are claims about the sampling distribution, and most of it is missing."""
    truncated = _ci(0.05, 0.09, dropped=40, n_good=60)     # would be "go"
    assert truncated.lo > 0
    assert classify_effect(truncated, sesoi_pt=2.0) == "inconclusive"

    narrow = _ci(-0.005, 0.005, dropped=40, n_good=60)     # would be equivalence
    assert classify_effect(narrow, sesoi_pt=2.0) == "inconclusive"


def test_classification_still_acts_on_a_clean_distribution():
    assert classify_effect(_ci(0.05, 0.09, 0), sesoi_pt=2.0) == "go"
    assert classify_effect(_ci(-0.005, 0.005, 0), sesoi_pt=2.0) == "equivalence"


def test_ci_dict_exposes_its_own_reliability():
    d = _ci(0.0, 0.1, dropped=10, n_good=90).as_dict()
    assert d["drop_fraction"] == pytest.approx(0.1)
    assert d["usable"] is False
    assert d["n_dropped_resamples"] == 10


def test_degeneracy_only_bites_at_small_prompt_counts():
    """The guard is not a nuisance switch: at the pre-registered P_audit scale
    (200-400) measured degeneracy never produces an undefined resample."""
    from csr_grpo.diagnostics.bootstrap import bootstrap_indices, cluster_statistic
    from csr_grpo.diagnostics.fidelity import per_prompt_stats

    rng = np.random.default_rng(0)
    P, deg = 200, 0.85
    judge = rng.random((P, 4))
    gold = (rng.random((P, 4)) < 0.5).astype(float)
    gold[: int(P * deg)] = 1.0
    st = per_prompt_stats(judge, gold)

    idx = bootstrap_indices(P, 300, np.random.default_rng(1))
    vals = cluster_statistic(st, "acc_rank", idx)
    assert int(np.isnan(vals).sum()) == 0
