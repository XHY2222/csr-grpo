"""ΔÂ_canon semantics, the compatibility matrix, and the collapse guard."""

import numpy as np
import pytest

from csr_grpo.diagnostics.canonicalization import (
    CODE_TRANSFORMS, IFEVAL_TRANSFORMS, MATH_TRANSFORMS, CanonDetector,
    admissible, compute_canon, evaluate_warning, label_preservation,
    select_transform, verify_code_transform,
)


# ---------------------------------------------------------------------------
# the mathematical property quoted in the paper
# ---------------------------------------------------------------------------


def test_constant_shift_gives_zero_delta_adv():
    """A rewrite that adds a constant to every score must be invisible."""
    rng = np.random.default_rng(0)
    original = rng.random((20, 8))
    rewritten = original + 0.25          # same constant across the whole group
    rep = compute_canon(original, rewritten)
    assert rep.delta_adv == pytest.approx(0.0, abs=1e-9)
    assert rep.delta_raw == pytest.approx(0.25, abs=1e-9)   # raw does move
    assert rep.flip_rate == pytest.approx(0.0)


def test_reordering_gives_positive_delta_and_flips():
    rng = np.random.default_rng(1)
    original = rng.random((20, 8))
    rewritten = original[:, ::-1]        # reverse within each group
    rep = compute_canon(original, rewritten)
    assert rep.delta_adv > 0
    assert rep.flip_rate > 0


def test_identical_scores_give_zero():
    a = np.random.default_rng(2).random((10, 6))
    rep = compute_canon(a, a)
    assert rep.delta_adv == 0.0
    assert rep.flip_rate == 0.0


# ---------------------------------------------------------------------------
# denominator-collapse guard
# ---------------------------------------------------------------------------


def test_low_variance_groups_are_excluded_and_counted():
    flat = np.full((5, 8), 0.5)          # σ_r = 0 -> excluded
    noisy = np.random.default_rng(3).random((5, 8))
    original = np.vstack([flat, noisy])
    rewritten = original.copy()
    rewritten[:5] += 0.01
    rep = compute_canon(original, rewritten, sigma_min=1e-3)
    assert rep.n_excluded_small_sigma == 5
    assert rep.n_groups == 5
    assert rep.coverage == 0.5


def test_interpretation_warns_when_coverage_is_low():
    flat = np.full((10, 8), 0.5)
    rep = compute_canon(flat, flat + 0.01)
    assert "coverage" in rep.interpretation
    assert not np.isfinite(rep.delta_adv)


# ---------------------------------------------------------------------------
# compatibility matrix (operator x constraint)
# ---------------------------------------------------------------------------


def test_word_adding_transform_blocked_by_count_constraints():
    t = next(t for t in IFEVAL_TRANSFORMS if t.name == "polite_prefix")
    assert not admissible(t, ["word_count"])
    assert not admissible(t, ["char_count"])
    assert admissible(t, ["format_only"])


def test_whitespace_transform_survives_counting_but_not_blank_line_rules():
    t = next(t for t in IFEVAL_TRANSFORMS if t.name == "blank_lines")
    assert admissible(t, ["word_count"])       # word count unchanged
    assert not admissible(t, ["no_blank_lines"])


def test_select_transform_falls_back_or_excludes():
    assert select_transform("ifeval", ["word_count"]).name == "blank_lines"
    # every operator blocked -> no admissible rewrite, prompt excluded
    assert select_transform("ifeval", ["word_count", "no_blank_lines"]) is None


def test_code_and_math_transforms_never_touch_semantics_constraints():
    for t in (*CODE_TRANSFORMS, *MATH_TRANSFORMS):
        assert admissible(t, ["anything"])


# ---------------------------------------------------------------------------
# transforms themselves
# ---------------------------------------------------------------------------


def test_math_transform_only_swaps_transition_words():
    src = "Step 1: compute 3+4.\nTherefore, the answer is 7."
    out = MATH_TRANSFORMS[0](src)
    assert "3+4" in out and "7" in out
    assert out != src
    for ch in "0123456789+-*/=<>":
        assert out.count(ch) == src.count(ch), ch


def test_code_comment_drop_keeps_code_intact():
    src = "# header\nx = compute(1)\n# trailing"
    out = next(t for t in CODE_TRANSFORMS if t.name == "drop_comment")(src)
    assert out.strip() == "x = compute(1)"


def test_code_invariance_check_requires_a_runner():
    r = verify_code_transform("x=1", "x=1", samples=[""])
    assert r["verified"] is False and "runner" in r["reason"]


def test_code_invariance_detects_mismatch():
    r = verify_code_transform("print(1)", "print(2)", samples=["", ""],
                              run=lambda code, inp: code)
    assert r["verified"] is False and r["n_mismatch"] == 2


def test_code_invariance_passes_when_identical():
    r = verify_code_transform("print(1)", "print(1)", samples=["a", "b"],
                              run=lambda code, inp: "1")
    assert r["verified"] is True


# ---------------------------------------------------------------------------
# calibration and warning evaluation
# ---------------------------------------------------------------------------


def test_detector_calibrates_on_anchors_and_needs_consecutive_hits():
    det = CanonDetector.calibrate([0.01, 0.012, 0.011, 0.009], k=2.0, consecutive=2)
    assert det.threshold > det.mu        # mu + 2σ, not an absolute level
    assert det.first_trigger([0.0, 0.0, 0.0]) is None
    assert det.first_trigger([0.10, 0.0, 0.10, 0.10]) == 3   # needs two in a row


def test_warning_lead_and_lag():
    assert evaluate_warning(trigger_step=50, peak_step=120, horizon=200)["lead"] == 70
    late = evaluate_warning(trigger_step=150, peak_step=120, horizon=200)
    assert late["status"] == "late" and late["false_positive"] is True
    assert evaluate_warning(None, 120, 200)["false_negative"] is True


def test_label_preservation_rate():
    # label_fn keys on the first character: "a"->a, "z"->z, so only pair 1 changes
    out = label_preservation(["a1", "b1"], ["a2", "z2"], label_fn=lambda s: s[0])
    assert out["preservation_rate"] == 0.5
    assert out["changed"] == [1]


def test_label_preservation_counts_identical_pairs_as_preserved():
    out = label_preservation(["a", "b"], ["a", "b"], label_fn=lambda s: s)
    assert out["preservation_rate"] == 1.0 and out["changed"] == []
