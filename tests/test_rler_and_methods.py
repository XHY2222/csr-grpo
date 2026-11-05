"""RLER aggregation maths and the method plans (decoder of the paper's matrix)."""

import numpy as np
import pytest

from csr_grpo.config.schema import Config
from csr_grpo.methods.base import make_method, plan_table
from csr_grpo.methods.rler import (
    AdaptiveInterpolation, RLERFaithfulMethod, aggregate_distributions,
    disagreement, disagreement_aware_select, ensemble_scores, rler_rewards,
)


# ---------------------------------------------------------------------------
# RLER maths
# ---------------------------------------------------------------------------


def test_aggregate_distributions_is_weighted_mean():
    p = np.array([[0.7, 0.3, 0.0], [0.1, 0.4, 0.5]])
    agg = aggregate_distributions(p)
    assert agg.tolist() == pytest.approx([0.4, 0.35, 0.25])
    assert agg.sum() == pytest.approx(1.0)


def test_aggregate_rejects_unnormalised_rows():
    with pytest.raises(ValueError):
        aggregate_distributions(np.array([[0.5, 0.4], [0.5, 0.5]]))


def test_disagreement_zero_when_identical_and_max_when_disjoint():
    assert disagreement(np.array([[0.5, 0.5], [0.5, 0.5]])) == pytest.approx(0.0, abs=1e-9)
    assert disagreement(np.array([[1.0, 0.0], [0.0, 1.0]])) == pytest.approx(1.0, abs=1e-6)


def test_adaptive_weight_is_monotone_in_disagreement():
    a = AdaptiveInterpolation(k=4.0, midpoint=0.25)
    assert a.weight(0.0) < a.weight(0.25) < a.weight(0.9)
    assert a.weight(0.25) == pytest.approx(0.5)


def test_interpolation_endpoints():
    a = AdaptiveInterpolation(k=40.0, midpoint=0.25)   # steep gate: approach limits
    soft, hard = np.array([0.3, 0.3, 0.4]), np.array([0.0, 0.0, 1.0])
    assert a.mix_with(soft, hard, js=0.0) == pytest.approx(hard, abs=1e-3)     # agree -> hard
    assert a.mix_with(soft, hard, js=1.0) == pytest.approx(soft, abs=1e-3)     # disagree -> soft


def test_interpolation_default_gate_is_gentle():
    """With the default gate, agreement 0 still carries some soft weight."""
    a = AdaptiveInterpolation()
    soft, hard = np.array([0.3, 0.3, 0.4]), np.array([0.0, 0.0, 1.0])
    mixed = a.mix_with(soft, hard, js=0.0)
    assert not np.allclose(mixed, hard) and not np.allclose(mixed, soft)


def test_rler_rewards_rewards_the_consensus_answer():
    agg = np.array([0.1, 0.7, 0.2])
    interp = AdaptiveInterpolation()
    r = rler_rewards(np.array([1, 0, 2]), agg, interp=interp, js=0.0)
    assert r[0] > r[2] > r[1]


def test_disagreement_aware_select_keeps_requested_fraction():
    rng = np.random.default_rng(0)
    lp = rng.normal(size=(20, 2))
    idx = disagreement_aware_select(lp, 0.5, rng)
    assert 8 <= len(idx) <= 12
    assert len(set(idx.tolist())) == len(idx)


def test_ensemble_scores_mean():
    s = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    assert ensemble_scores(s).tolist() == [[0.5, 0.5]]


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------


def test_rler_matches_rollout_budget():
    cfg = Config()
    cfg.train.group_size = 8
    m = RLERFaithfulMethod.from_config(cfg)
    assert m.per_policy_group == 4
    plan = m.plan()
    assert plan.n_subpolicies == 2
    assert "matched budget" in " ".join(plan.notes)


def test_rler_rejects_indivisible_budget():
    m = RLERFaithfulMethod(group_size=9, n_subpolicies=2)
    with pytest.raises(ValueError):
        _ = m.per_policy_group


def test_method_registry_covers_matrix():
    names = ["shared_grpo", "dual_adapter", "dual_adapter_replay",
             "rler_faithful", "judge_lora_ensemble", "fisher_mask",
             "random_mask", "gopo_rank_only", "gold_reference"]
    rows, methods = [], []
    for n in names:
        cfg = Config()
        cfg.method = n
        cfg.is_reference = n == "gold_reference"
        cfg.validate()
        m = make_method(cfg)
        methods.append(m)
        rows.append(m.plan())
    table = plan_table(methods)
    assert len(table) == len(names)
    shared = rows[0]
    dual = rows[1]
    assert shared.spec.r_shared == 32 and shared.judge_trainable_in_stage
    assert dual.spec.r_shared == 0 and not dual.judge_trainable_in_stage
    assert rows[2].replay
    assert rows[5].mask_kind == "fisher"
    assert rows[6].mask_kind == "random"
    assert rows[7].adv_mode == "rank_only"
    assert rows[8].uses_gold_reward


def test_gold_reward_requires_reference_or_diagnostic_flag():
    cfg = Config()
    cfg.reward_source = "gold"
    with pytest.raises(ValueError, match="supervision ledger"):
        cfg.validate()
    cfg.is_diagnostic_arm = True
    cfg.validate()


def test_double_dip_config_is_rejected():
    cfg = Config()
    cfg.method = "dual_adapter_replay"
    cfg.stages_to_run = (1,)
    with pytest.raises(ValueError):
        cfg.validate()
