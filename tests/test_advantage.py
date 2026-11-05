"""GRPO / rank-only / degenerate-zeroing, including the plan's worked GOPO example."""

import numpy as np
import pytest

from csr_grpo.rewards.advantage import (
    AdvantageConfig, compute_advantages, degenerate_mask_from_gold,
    effective_sample_weight, rank_transform,
)
from csr_grpo.supervision.ledger import Role, SupervisionLedger, SupervisionViolation


def test_rank_transform_average_ties():
    r = rank_transform(np.array([[0.5, 0.1, 0.1, 0.9]]))
    assert r[0].tolist() == [3.0, 1.5, 1.5, 4.0]


def test_gopo_example_from_the_plan():
    """gold [1,0,0,0] with the correct answer ranked first.

    Rank scores become [4,3,2,1]; subtracting the mean 2.5 leaves
    [+1.5,+0.5,-0.5,-1.5] — the second *wrong* answer still gets a positive
    advantage.  This is why rank-only is an ablation about magnitude
    distortion, not a fix for ranking errors.
    """
    gold = np.array([[1.0, 0.0, 0.0, 0.0]])
    judge = np.array([[0.99, 0.60, 0.59, 0.58]])
    adv_rank = compute_advantages(judge, cfg=AdvantageConfig(mode="rank_only", eps=0.0))
    z = (np.array([4.0, 3.0, 2.0, 1.0]) - 2.5) / np.std([4, 3, 2, 1])
    assert adv_rank[0] == pytest.approx(z, abs=1e-6)
    assert adv_rank[0, 1] > 0


def test_grpo_advantages_sum_to_zero():
    rng = np.random.default_rng(0)
    r = rng.random((16, 8))
    a = compute_advantages(r)
    assert np.allclose(a.sum(axis=1), 0.0, atol=1e-6)


def test_degenerate_zeroing_removes_off_gold_contrast():
    """Arm (iv): a group the judge calls all-good gets its invented contrast zeroed."""
    judge = np.array([[0.95, 0.96, 0.97, 0.94]])   # all >= 0.9 -> judge-degenerate
    raw = compute_advantages(judge)
    assert np.abs(raw).max() > 0.5                 # judge invents a contrast
    zeroed = compute_advantages(
        judge, cfg=AdvantageConfig(degenerate_zero=True, degenerate_source="judge"))
    assert np.allclose(zeroed, 0.0)


def test_degenerate_zeroing_leaves_informative_groups_alone():
    judge = np.array([[0.95, 0.96, 0.97, 0.94, 0.93, 0.92, 0.91, 0.90],  # all high
                       [0.90, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01]])  # spread
    zeroed = compute_advantages(
        judge, cfg=AdvantageConfig(degenerate_zero=True, degenerate_source="judge"))
    assert np.allclose(zeroed[0], 0.0)
    assert np.abs(zeroed[1]).max() > 0.5


def test_gold_mask_requires_ledger_and_is_blocked_in_training():
    led = SupervisionLedger()
    gold = np.array([[1.0, 1.0], [1.0, 0.0]])
    judge = np.zeros((2, 2))
    with led.training_scope(), pytest.raises(SupervisionViolation):
        compute_advantages(judge, cfg=AdvantageConfig(degenerate_zero=True,
                                                      degenerate_source="gold"),
                           gold=gold, ledger=led)
    # outside a training scope the same call is an audited diagnostic read
    out = compute_advantages(judge, cfg=AdvantageConfig(degenerate_zero=True,
                                                        degenerate_source="gold"),
                             gold=gold, ledger=led)
    assert out.shape == gold.shape


def test_clip_bounds():
    r = np.array([[0.0, 0.0, 0.0, 1.0]])
    a = compute_advantages(r, cfg=AdvantageConfig(clip=1.0))
    assert np.abs(a).max() <= 1.0


def test_mask_from_gold_all_or_nothing():
    gold = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 0]])
    assert degenerate_mask_from_gold(gold).tolist() == [True, True, False]


def test_effective_weight_is_zero_for_degenerate():
    adv = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
    w = effective_sample_weight(adv)
    assert np.allclose(w[0], 0.0)
    assert pytest.approx(float(w[1].sum())) == 1.0
