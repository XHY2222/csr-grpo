"""Rows 5b / 5c / 6 — the three co-rewarding baselines.

Each is a *decisive baseline* in the plan's run matrix, so "it constructs
without raising" would not be a test: the numbers they contribute to the main
table come from the preference-pair filter, the replay flag, and the EMA lag +
entropy crop.  Those are the things asserted here.
"""

import numpy as np
import pytest
import torch

from csr_grpo.config.schema import Config
from csr_grpo.methods.base import make_method
from csr_grpo.methods.co_rewarding import (
    EMAJudge,
    MGRPOAdaptedMethod,
    MetaRewardingMethod,
    OnlineJudgeCoTrainingMethod,
    apply_entropy_filter,
    entropy_filter_mask,
    meta_preference_loss,
    meta_preference_pairs,
)


# ---------------------------------------------------------------------------
# Row 5b — Meta-Rewarding preference construction
# ---------------------------------------------------------------------------


def test_meta_pairs_only_survive_meta_disagreement():
    # judge says a > b > c; meta-judge reverses everything.
    s = np.array([3.0, 2.0, 1.0])
    m = np.array([0.0, 5.0, 9.0])
    pairs = meta_preference_pairs(s, m)
    # every meta-ordered pair is one the judge got backwards
    assert set(pairs) == {(1, 0), (2, 1), (2, 0)}


def test_meta_pairs_drop_consistent_and_tied_judgements():
    s = np.array([3.0, 2.0, 1.0])
    m = np.array([3.0, 2.0, 1.0])          # meta agrees with judge everywhere
    assert meta_preference_pairs(s, m) == []

    m_tie = np.array([5.0, 5.0, 5.0])      # no meta information at all
    assert meta_preference_pairs(s, m_tie) == []


def test_meta_pairs_reject_misaligned_vectors():
    with pytest.raises(ValueError):
        meta_preference_pairs(np.zeros(3), np.zeros(4))


def test_meta_loss_is_zero_without_pairs_and_decreasing_with_agreement():
    scores = torch.tensor([1.0, 2.0], requires_grad=True)
    empty = meta_preference_loss(scores, [])
    assert float(empty.detach()) == 0.0
    assert empty.requires_grad

    # winner index 0 (score 1.0) is *below* loser index 1 (score 2.0): the
    # Bradley-Terry term is large and pushes the ordering to flip.
    wrong = meta_preference_loss(scores, [(0, 1)])
    right = meta_preference_loss(torch.tensor([2.0, 1.0]), [(0, 1)])
    assert float(wrong) > float(right)
    # analytic value: -log sigmoid(-1) = log(1 + e^1)
    assert float(wrong) == pytest.approx(float(np.log1p(np.e)), rel=1e-5)


def test_meta_rewarding_plan_trains_the_judge_in_stage():
    cfg = Config()
    cfg.method = "meta_rewarding"
    m = make_method(cfg)
    assert isinstance(m, MetaRewardingMethod)
    p = m.plan()
    assert p.judge_trainable_in_stage is True
    # the paper's mechanism is single-stage: no replay, and the run matrix holds
    # the architecture fixed at disjoint (Row 3 is the shared-architecture arm),
    # so 5b must not also vary r_shared or the comparison confounds two factors.
    assert p.replay is False
    assert p.spec.r_shared == 0
    assert p.spec.r_actor_private > 0


# ---------------------------------------------------------------------------
# Row 5c — online judge co-training
# ---------------------------------------------------------------------------


def test_co_training_plan_is_labelled_as_an_adaptation():
    cfg = Config()
    cfg.method = "online_judge_cotraining"
    m = make_method(cfg)
    assert isinstance(m, OnlineJudgeCoTrainingMethod)
    p = m.plan()
    assert p.judge_trainable_in_stage is True
    assert p.replay is True              # history pairs are the other half
    text = " ".join(p.notes).lower()
    assert "our adaptation" in text, "the plan requires the provenance label"


def test_co_training_differs_from_row8_on_both_axes():
    """5c vs 8 differs in judge training *and* replay; asserting both keeps the
    ablation honest if either flag is later flipped by accident."""
    cfg = Config()
    cfg.method = "online_judge_cotraining"
    five_c = make_method(cfg).plan()
    cfg.method = "dual_adapter_replay"
    row8 = make_method(cfg).plan()
    assert five_c.judge_trainable_in_stage and not row8.judge_trainable_in_stage
    assert five_c.replay is True


# ---------------------------------------------------------------------------
# Row 6 — EMA judge + entropy filter
# ---------------------------------------------------------------------------


def test_ema_judge_is_identity_on_first_update():
    ema = EMAJudge(tau=0.995)
    params = {"judge.B": torch.tensor([1.0, 2.0])}
    ema.update(params)
    assert ema.initialized
    assert torch.equal(ema.shadow["judge.B"], params["judge.B"])
    assert ema.lens(params) == pytest.approx(0.0)


def test_ema_judge_moves_at_one_minus_tau():
    ema = EMAJudge(tau=0.9)
    ema.update({"w": torch.tensor([0.0])})
    ema.update({"w": torch.tensor([1.0])})
    assert float(ema.shadow["w"]) == pytest.approx(0.1, rel=1e-6)
    ema.update({"w": torch.tensor([1.0])})
    assert float(ema.shadow["w"]) == pytest.approx(0.19, rel=1e-6)


def test_ema_lens_is_the_student_teacher_distance():
    ema = EMAJudge(tau=1.0)               # shadow never moves: pure lag readout
    ema.update({"w": torch.tensor([0.0, 0.0])})
    assert ema.lens({"w": torch.tensor([3.0, 4.0])}) == pytest.approx(5.0)


def test_mgrpo_plan_reads_ema_tau_from_config():
    cfg = Config()
    cfg.method = "mgrpo_adapted"
    cfg.judge.ema_tau = 0.99
    m = make_method(cfg)
    assert isinstance(m, MGRPOAdaptedMethod)
    assert m.tau == pytest.approx(0.99)
    assert m.ema_judge().tau == pytest.approx(0.99)
    assert m.plan().judge_trainable_in_stage is True


def test_entropy_filter_keeps_the_confident_trajectories():
    """The filter's whole purpose is to drop noisy (high-entropy) rollouts; a
    descending sort would keep exactly those."""
    confident = torch.full((1, 4), -0.01)     # near-deterministic
    uniform = torch.full((1, 4), -2.7726)     # uniform over 16 tokens
    muddled = torch.full((1, 4), -1.5)
    lp = torch.cat([confident, uniform, muddled])
    keep = entropy_filter_mask(lp, keep_fraction=1 / 3)
    assert keep.tolist() == [True, False, False]

    keep_two = entropy_filter_mask(lp, keep_fraction=2 / 3)
    assert keep_two.tolist() == [True, False, True]


def test_entropy_filter_respects_the_requested_fraction():
    lp = -torch.rand(10, 5) * 3.0
    for frac in (0.1, 0.25, 0.5, 1.0):
        keep = entropy_filter_mask(lp, keep_fraction=frac)
        assert int(keep.sum()) == max(1, round(frac * 10))


def test_entropy_filter_validates_its_inputs():
    with pytest.raises(ValueError):
        entropy_filter_mask(torch.zeros(4), keep_fraction=0.5)   # not (N, T)
    with pytest.raises(ValueError):
        entropy_filter_mask(-torch.rand(4, 2), keep_fraction=0.0)


def test_apply_entropy_filter_zeroes_only_the_dropped_rows():
    adv = np.array([1.0, 2.0, 3.0, 4.0])
    keep = torch.tensor([True, False, True, False])
    out = apply_entropy_filter(adv, keep)
    assert out.tolist() == [1.0, 0.0, 3.0, 0.0]
    # the input is not mutated
    assert adv.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_apply_entropy_filter_works_on_grouped_advantages():
    adv = np.arange(6, dtype=float).reshape(3, 2)
    out = apply_entropy_filter(adv, torch.tensor([True, False, True]))
    assert out[1].tolist() == [0.0, 0.0]
    assert out[0].tolist() == [0.0, 1.0]


def test_apply_entropy_filter_rejects_a_mismatched_mask():
    with pytest.raises(ValueError):
        apply_entropy_filter(np.zeros(4), torch.tensor([True, False]))
