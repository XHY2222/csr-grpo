"""Metric definitions, including the degenerate-group rules that are easy to get wrong."""

import numpy as np
import pytest

from csr_grpo.diagnostics.fidelity import (
    acc_rank_group, advantage_mse, advantages, aggregate, evaluate_line,
    fisher_z_mean, mse_rho_identity, per_prompt_stats, rho_group,
    sign_agree_group,
)


def test_acc_rank_perfect_and_inverted():
    g = np.array([1.0, 0.0, 0.0, 0.0])
    assert acc_rank_group(np.array([0.9, 0.1, 0.2, 0.3]), g) == 1.0
    assert acc_rank_group(np.array([0.1, 0.9, 0.2, 0.3]), g) == 0.0


def test_acc_rank_ties_get_half_credit():
    g = np.array([1.0, 0.0])
    assert acc_rank_group(np.array([0.5, 0.5]), g, tie="half") == 0.5
    assert acc_rank_group(np.array([0.5, 0.5]), g, tie="strict") == 0.0


def test_acc_rank_undefined_on_degenerate_groups_not_zero():
    assert acc_rank_group(np.array([0.9, 0.8]), np.array([1.0, 1.0])) is None
    assert acc_rank_group(np.array([0.9, 0.8]), np.array([0.0, 0.0])) is None


def test_sign_agree_catches_magnitude_distortion():
    """Plan F1's counterexample: perfect ranking, wrong sign for the 2nd correct answer."""
    gold = np.array([[1.0, 1.0, 0.0, 0.0]])
    judge = np.array([[0.99, 0.60, 0.59, 0.58]])
    aj, ag = advantages(judge), advantages(gold)
    assert acc_rank_group(judge[0], gold[0]) == 1.0     # ranking is perfect
    # gold advantages are [1, 1, -1, -1]; judge gives the 2nd correct answer a
    # *negative* advantage, so 3 of 4 signs agree - worse than the ranking.
    # σ(ε) = 1 + 1e-4 shrink the magnitudes slightly; signs are exact
    assert np.sign(ag[0]).tolist() == [1.0, 1.0, -1.0, -1.0]
    assert ag[0].tolist() == pytest.approx([1.0, 1.0, -1.0, -1.0], abs=1e-3)
    assert aj[0, 1] < 0
    assert sign_agree_group(aj[0], ag[0]) == 0.75
    assert sign_agree_group(aj[0], ag[0]) < acc_rank_group(judge[0], gold[0])


def test_rho_adv_and_mse_identity():
    rng = np.random.default_rng(0)
    gold = (rng.random((50, 8)) > 0.5).astype(float)
    judge = gold * 0.8 + rng.random((50, 8)) * 0.2
    ident = mse_rho_identity(judge, gold)
    # z-scoring gives MSE = 2(1-rho) up to the eps stabiliser
    assert abs(ident["gap"]) < 5e-3


def test_pseudo_contrast_only_on_degenerate_groups():
    gold = np.array([
        [1.0, 1.0, 1.0, 1.0],     # degenerate (all correct)
        [0.0, 0.0, 0.0, 0.0],     # degenerate (all wrong)
        [1.0, 0.0, 0.0, 0.0],     # informative
    ])
    judge = np.array([
        [0.9, 0.5, 0.4, 0.1],     # spread 0.8 > tau  -> pseudo contrast
        [0.2, 0.21, 0.22, 0.23],  # spread 0.03 < tau -> no contrast
        [0.9, 0.1, 0.2, 0.3],
    ])
    st = per_prompt_stats(judge, gold, tau=0.05)
    assert st.degenerate.tolist() == [True, True, False]
    assert st.pseudo_contrast[st.degenerate].tolist() == [True, False]
    assert aggregate(st, "pseudo_contrast") == 0.5
    assert st.valid_pair_coverage == pytest.approx(1 / 3)


def test_degenerate_groups_excluded_from_means():
    gold = np.array([[1.0, 1.0], [1.0, 0.0]])
    judge = np.array([[0.9, 0.1], [0.9, 0.1]])
    st = per_prompt_stats(judge, gold)
    assert np.isnan(st.acc_rank[0]) and st.acc_rank[1] == 1.0
    assert aggregate(st, "acc_rank") == 1.0


def test_fisher_z_mean_matches_plain_mean_when_equal_n():
    rs = [0.2, -0.1, 0.5]
    got = fisher_z_mean(rs, [8, 8, 8])
    plain = float(np.mean(rs))
    assert abs(got - plain) < 0.05   # z-space shrinkage is small, not absent


def test_evaluate_line_reports_coverage():
    rng = np.random.default_rng(1)
    gold = (rng.random((20, 8)) > 0.7).astype(float)
    judge = gold * 0.9 + rng.random((20, 8)) * 0.1
    rep = evaluate_line(judge, gold)
    assert rep.n_prompts == 20
    assert rep.n_informative + rep.n_degenerate == 20
    assert 0.0 <= rep.values["valid_pair_coverage"] <= 1.0


def test_advantages_normalisation():
    a = advantages(np.array([[1.0, 2.0, 3.0, 4.0]]))
    assert abs(float(a.mean())) < 1e-6
    assert a[0, 3] > 0 > a[0, 0]
