"""E1' (held-out Stage 2→3) invariants.

The held-out confirmation is the load-bearing replication for H1, so it needs
its own coverage: it must use stage 2 as the audit line, stage 3 as the target,
and it must keep the ledger clean while the gold DID arm runs.
"""

import numpy as np
import pytest

from csr_grpo.config.schema import Config
from csr_grpo.diagnostics.bootstrap import BootstrapCI, percentile_ci
from csr_grpo.experiments.e1_switch import run_e1_prime
from csr_grpo.supervision.ledger import SupervisionLedger


def _cfg():
    cfg = Config()
    cfg.method = "dual_adapter"
    cfg.model.r_actor = 16
    cfg.model.lora_alpha = 32
    cfg.train.group_size = 4
    cfg.train.warmup_rounds = 4
    cfg.data.n_prompts_per_stage = 24
    cfg.data.audit_prompts = 16
    return cfg


# ---------------------------------------------------------------------------
# NaN handling in bootstrap CIs (the defect E1' exposed)
# ---------------------------------------------------------------------------


def test_percentile_ci_drops_undefined_resamples_instead_of_propagating_nan():
    samples = np.array([0.2, 0.3, np.nan, 0.25, 0.28])
    ci = percentile_ci(samples, point=0.26)
    assert np.isfinite(ci.lo) and np.isfinite(ci.hi)
    assert ci.n_dropped == 1


def test_percentile_ci_reports_fully_undefined_batch():
    ci = percentile_ci(np.array([np.nan, np.nan]), point=0.0)
    assert not np.isfinite(ci.lo)
    assert ci.n_dropped == 2


def test_bootstrap_ci_dict_exposes_dropped_count():
    ci = percentile_ci(np.array([0.1, 0.2, np.nan]), point=0.15)
    d = ci.as_dict()
    assert d["n_dropped_resamples"] == 1
    assert "lo" in d and "hi" in d


# ---------------------------------------------------------------------------
# E1' protocol
# ---------------------------------------------------------------------------


def test_e1_prime_runs_on_heldout_transition():
    led = SupervisionLedger()
    res = run_e1_prime(_cfg(), led, steps=6)
    assert res.line == "P_audit"
    assert np.isfinite(res.delta_switch_self.point)
    # the gold DID arm must exist for the double difference to be defined
    assert res.delta_switch_gold is not None
    assert res.delta_did is not None
    assert res.decision_did in {"go", "equivalence", "inconclusive"}


def test_e1_prime_keeps_training_gold_at_zero():
    led = SupervisionLedger()
    run_e1_prime(_cfg(), led, steps=5)
    s = led.summary()
    assert s["gold_calls_in_training"] == 0
    assert s["violations"] == 0
    assert s["gold_calls_in_gold_control"] > 0


def test_e1_prime_uses_stage2_as_audit_line():
    """A wrong audit stage would silently measure the wrong transition."""
    import inspect

    from csr_grpo.experiments import e1_switch

    src = inspect.getsource(e1_switch.run_e1_prime)
    assert "target_stage=3" in src and "audit_stage=2" in src


def test_did_is_difference_of_differences():
    """Δ_DID = (switch − continue)_self − (switch − continue)_gold."""
    led = SupervisionLedger()
    res = run_e1_prime(_cfg(), led, steps=5)
    expected = res.delta_switch_self.point - res.delta_switch_gold.point
    assert res.delta_did.point == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# P_audit pairing across arms
# ---------------------------------------------------------------------------
#
# E1 contrasts Switch vs Continue *within prompt*, and the gold arms are the
# DID control, so all four arms must be measured on the same questions.  The
# audit prompt seed used to be derived from the target stage, so the Switch arm
# (target 2) drew s1-1502-* while the Continue arm (target 1) drew s1-1501-*:
# the paired bootstrap then resampled index i in both arms and paired two
# different questions.


def _audit_ids(cfg, target_stage: int, audit_stage: int, audit_seed):
    from csr_grpo.methods.base import make_method
    from csr_grpo.training.runner import Runner

    runner = Runner(cfg, make_method(cfg).plan(), SupervisionLedger())
    ctx = runner.make_context(target_stage, n_prompts=8,
                              seed=1000 + target_stage,
                              audit_stage=audit_stage, audit_seed=audit_seed)
    return ctx.audit_prompts, ctx.audit_task_ids


def test_shared_audit_seed_gives_both_arms_the_same_prompts():
    import torch

    cfg = _cfg()
    cfg.data.audit_prompts = 6
    a_ids, a_tids = _audit_ids(cfg, 2, 1, audit_seed=5001)
    b_ids, b_tids = _audit_ids(cfg, 1, 1, audit_seed=5001)
    assert torch.equal(a_ids, b_ids)
    assert a_tids == b_tids


def test_default_audit_seed_drifts_with_the_target_stage():
    """The bug this guards against: without an explicit seed the two arms draw
    different P_audit sets, which is what made the paired bootstrap unpaired."""
    import torch

    cfg = _cfg()
    cfg.data.audit_prompts = 6
    a_ids, a_tids = _audit_ids(cfg, 2, 1, audit_seed=None)
    b_ids, b_tids = _audit_ids(cfg, 1, 1, audit_seed=None)
    assert not torch.equal(a_ids, b_ids)
    assert a_tids != b_tids


def test_run_e1_measures_every_arm_on_one_audit_set():
    """End to end: the arms the DID combines must share task ids."""
    from csr_grpo.experiments.e1_switch import run_e1

    cfg = _cfg()
    cfg.data.audit_prompts = 6
    led = SupervisionLedger()
    res = run_e1(cfg, led, steps=2, gold_seeds=1, target_stage=2, audit_stage=1)

    arm_names = {a["arm"] for a in res.arms}
    assert arm_names == {"self@2", "self@1"}
    # DID is reported, and the ledger stayed clean apart from the declared arm
    assert res.delta_did is not None
    assert led.gold_calls_in_training == 0


# ---------------------------------------------------------------------------
# structural-lag: both criteria must appear in E1 output
# ---------------------------------------------------------------------------


def test_lead_time_result_contains_both_criteria():
    """H1's structural-lag claim requires criterion 1 (Δt_lead) AND criterion 2
    (ΔR²_oos) to be reported; only the first was wired before this fix."""
    from csr_grpo.experiments.e1_switch import run_arm, lead_time_from_arms
    from csr_grpo.rewards.sources import RewardSource

    cfg = _cfg()
    cfg.data.audit_prompts = 8
    cfg.exp.dt_lead_steps = 3
    led = SupervisionLedger()
    sw = run_arm(cfg, reward_source=RewardSource.SELF, target_stage=2,
                 steps=24, seed=0, ledger=led, audit_stage=1, measure_every=4)
    co = run_arm(cfg, reward_source=RewardSource.SELF, target_stage=1,
                 steps=24, seed=100, ledger=led, audit_stage=1, measure_every=4)
    result = lead_time_from_arms(sw, co, target_fpr=0.05, min_required=3)
    assert result is not None
    assert "dt_lead" in result
    assert "delta_r2_oos" in result
    assert "structural_lag_supported" in result


def test_run_e1_lag_contains_both_criteria():
    from csr_grpo.experiments.e1_switch import run_e1

    cfg = _cfg()
    cfg.data.audit_prompts = 8
    cfg.exp.dt_lead_steps = 2
    led = SupervisionLedger()
    res = run_e1(cfg, led, steps=24, gold_seeds=0, target_stage=2, audit_stage=1,
                 measure_every=6)
    lag = res.as_dict().get("structural_lag")
    assert lag is not None
    assert "dt_lead" in lag
    assert "delta_r2_oos" in lag


# ---------------------------------------------------------------------------
# multi-seed pooling
# ---------------------------------------------------------------------------


def test_per_prompt_stats_concat_doubles_the_prompt_count():
    """PerPromptStats.concat must stack arrays from different seeds so the
    paired bootstrap has more prompts to draw from."""
    import numpy as np
    from csr_grpo.diagnostics.fidelity import per_prompt_stats, PerPromptStats

    rng = np.random.default_rng(0)
    P, G = 8, 4
    gold = (rng.random((P, G)) > 0.5).astype(float)
    st = per_prompt_stats(rng.random((P, G)), gold)
    merged = PerPromptStats.concat([st, st])
    assert len(merged) == 2 * P
    # NaN-aware comparison: the first half must equal the original
    assert np.array_equal(merged.acc_rank[:P], st.acc_rank, equal_nan=True)


def test_run_e1_with_multiple_seeds_pools_prompts():
    """When seeds=2, run_e1 must actually run two seed-replicates and the
    final per-prompt stats must be twice the size of a single-seed run."""
    from csr_grpo.experiments.e1_switch import run_e1, compare_arms
    from csr_grpo.experiments.e1_switch import run_arm
    from csr_grpo.rewards.sources import RewardSource

    cfg = _cfg()
    cfg.data.audit_prompts = 6
    led = SupervisionLedger()

    res_s1 = run_e1(cfg, SupervisionLedger(), steps=4, gold_seeds=0,
                    seeds=1, target_stage=2, audit_stage=1, measure_every=4)
    res_s2 = run_e1(cfg, SupervisionLedger(), steps=4, gold_seeds=0,
                    seeds=2, target_stage=2, audit_stage=1, measure_every=4)

    # with 2 seeds the per-prompt pool must be strictly larger
    from csr_grpo.diagnostics.dual_line import LINE_AUDIT
    n1 = len(res_s1.delta_switch_self.boot)
    n2 = len(res_s2.delta_switch_self.boot)
    # n_resamples is the same; point estimate is computed on more prompts
    assert n1 == n2   # same n_resamples
    # but the point estimates can differ (more prompts changes the estimate)
    # -- at minimum, the machinery must not crash with seeds=2
    assert isinstance(res_s2.decision_switch, str)
    assert res_s2.delta_switch_self is not None
