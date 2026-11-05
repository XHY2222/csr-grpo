"""Dual measurement line, config loader, and reward-source stacks.

These three modules had no test referencing them at all, despite carrying
deliverable-1 (config system), deliverable-2 (dual line) and deliverable-3
(E2 reward arms) respectively.  The first thing the coverage found was two
real defects, both fixed here:

* ``DualLine.divergence`` documented "latest *common* step" but compared each
  line's own latest sample, so a partial run subtracted measurements from two
  different moments and reported the step gap as a line difference;
* ``--set train.lr=3e-5`` parsed as the *string* ``'3e-5'`` (PyYAML implements
  YAML 1.1, which wants a dot before the exponent), sailed through
  ``Config.validate``, and only failed inside ``torch.optim``.
"""

import numpy as np
import pytest

from csr_grpo.config.loader import _parse_scalar, apply_overrides, describe
from csr_grpo.config.schema import Config
from csr_grpo.diagnostics.dual_line import (
    LINE_AUDIT, LINE_ONPOLICY, DualLine, SnapshotBook,
)
from csr_grpo.rewards.advantage import AdvantageConfig
from csr_grpo.rewards.sources import (
    GoldRewardProvider, JudgeRewardProvider, RewardSource, build_arms,
)
from csr_grpo.supervision.ledger import (
    Level, Role, SupervisionLedger, SupervisionViolation,
)


# ---------------------------------------------------------------------------
# dual line
# ---------------------------------------------------------------------------


def _matrices(n=8, g=4, seed=0):
    rng = np.random.default_rng(seed)
    gold = (rng.random((n, g)) > 0.5).astype(float)
    judge = gold * 0.7 + rng.random((n, g)) * 0.3
    return judge, gold


def test_lines_are_kept_separate_and_aligned():
    judge, gold = _matrices()
    dl = DualLine()
    for step in (10, 20):
        dl.measure(LINE_AUDIT, step, judge, gold)
        dl.measure(LINE_ONPOLICY, step, judge * 0.5, gold)

    a_steps, a_vals = dl.series(LINE_AUDIT, "acc_rank")
    b_steps, b_vals = dl.series(LINE_ONPOLICY, "acc_rank")
    assert a_steps == b_steps == [10, 20]
    # line B's judge is a strictly worse ordering of the same gold, so its
    # discrimination metric cannot exceed line A's
    assert all(b <= a + 1e-12 for a, b in zip(a_vals, b_vals))


def test_unknown_line_is_rejected():
    judge, gold = _matrices()
    with pytest.raises(ValueError, match="unknown line"):
        DualLine().measure("P_something", 1, judge, gold)


def test_divergence_uses_the_latest_common_step():
    """Regression: comparing each line's own latest sample subtracts values
    from different steps and reports the gap as a line difference."""
    judge, gold = _matrices()
    dl = DualLine()
    dl.measure(LINE_AUDIT, 10, judge, gold)
    dl.measure(LINE_AUDIT, 12, judge * 0.2, gold)      # A ends at 12
    dl.measure(LINE_ONPOLICY, 10, judge, gold)
    dl.measure(LINE_ONPOLICY, 14, judge, gold)         # B ends at 14

    # the only step both lines measured is 10, where they are identical
    assert dl.divergence("acc_rank") == pytest.approx(0.0)


def test_divergence_is_undefined_without_a_shared_step():
    judge, gold = _matrices()
    dl = DualLine()
    dl.measure(LINE_AUDIT, 10, judge, gold)
    dl.measure(LINE_ONPOLICY, 12, judge, gold)
    assert np.isnan(dl.divergence())


def test_divergence_is_undefined_before_both_lines_exist():
    judge, gold = _matrices()
    dl = DualLine()
    dl.measure(LINE_AUDIT, 10, judge, gold)
    assert np.isnan(dl.divergence())


def test_report_counts_informative_and_degenerate_groups():
    n, g = 10, 4
    judge = np.random.default_rng(3).random((n, g))
    gold = np.ones((n, g))            # every group all-correct -> degenerate
    sample = DualLine().measure(LINE_AUDIT, 1, judge, gold)
    assert sample.report.n_degenerate == n
    assert sample.report.n_informative == 0
    # a degenerate-only line has no ranking denominator; it must not read 0.0
    assert np.isnan(sample.report.values["acc_rank"])


def test_table_exports_one_row_per_sample():
    judge, gold = _matrices()
    dl = DualLine()
    dl.measure(LINE_AUDIT, 1, judge, gold, note="a")
    dl.measure(LINE_ONPOLICY, 1, judge, gold, note="b")
    rows = dl.table()
    assert len(rows) == 2
    assert {r["line"] for r in rows} == {LINE_AUDIT, LINE_ONPOLICY}
    assert all("acc_rank" in r for r in rows)


# ---------------------------------------------------------------------------
# snapshot book (the 2x2 inputs)
# ---------------------------------------------------------------------------


def test_snapshot_book_completeness_tracks_filled_fields():
    book = SnapshotBook()
    assert not book.complete
    assert len(book.missing()) == 6

    arr = np.zeros((2, 2))
    for name in SnapshotBook._FIELDS[:-1]:
        book.set(name, arr)
    assert book.missing() == ["gold_on_yk"]

    book.set("gold_on_yk", arr)
    assert book.complete


def test_snapshot_book_rejects_unknown_field():
    with pytest.raises(KeyError):
        SnapshotBook().set("not_a_field", np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# config loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("3e-5", 3e-5),
    ("1e-5", 1e-5),
    ("1.0", 1.0),
    ("0.5", 0.5),
    ("0.5e-3", 0.0005),
    ("-2.5", -2.5),
])
def test_scientific_notation_parses_as_float(text, expected):
    out = _parse_scalar(text)
    assert isinstance(out, float)
    assert out == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [("8", 8), ("-2", -2), ("0", 0)])
def test_integers_stay_integers(text, expected):
    out = _parse_scalar(text)
    assert isinstance(out, int) and out == expected


def test_non_numeric_values_fall_back_to_yaml():
    assert _parse_scalar("true") is True
    assert _parse_scalar("dual_adapter") == "dual_adapter"
    assert _parse_scalar("[1, 2]") == [1, 2]


def test_override_reaches_the_dataclass_as_a_float():
    cfg = apply_overrides(Config(), ["train.lr=3e-5"])
    assert isinstance(cfg.train.lr, float)
    assert cfg.train.lr == pytest.approx(3e-5)


def test_unknown_override_paths_are_rejected():
    with pytest.raises(ValueError, match="unknown section"):
        apply_overrides(Config(), ["nosuch.lr=1.0"])
    with pytest.raises(ValueError, match="unknown key"):
        apply_overrides(Config(), ["train.nosuch=1.0"])
    with pytest.raises(ValueError, match="must look like"):
        apply_overrides(Config(), ["train.lr"])


def test_validate_rejects_a_numeric_field_holding_a_string():
    """The failure mode that reached torch.optim: every range check compares a
    string and short-circuits, so the config looked valid."""
    cfg = Config()
    cfg.train.lr = "3e-5"
    with pytest.raises(ValueError, match="train.lr must be numeric"):
        cfg.validate()


def test_validate_accepts_the_parsed_float():
    cfg = apply_overrides(Config(), ["train.lr=3e-5"])
    assert cfg.validate() is cfg


def test_validate_rejects_a_string_onnx_like_field_too():
    """Guards the general rule, not just the lr that happened to surface it."""
    cfg = Config()
    cfg.exp.delta_sesoi_pt = "2.0"
    with pytest.raises(ValueError, match="exp.delta_sesoi_pt must be numeric"):
        cfg.validate()


def test_describe_names_the_gold_posture():
    cfg = Config()
    assert "trained" in describe(cfg)
    cfg.is_reference = True
    assert "reference" in describe(cfg)
    cfg.is_reference = False
    cfg.is_diagnostic_arm = True
    assert "diagnostic" in describe(cfg)


# ---------------------------------------------------------------------------
# reward sources
# ---------------------------------------------------------------------------


def _judge(task_id, answers):
    # deterministic: score falls with the answer index, so ranking is exact
    return [1.0 - 0.1 * i for i in range(len(answers))]


def test_judge_provider_returns_a_score_matrix():
    prov = JudgeRewardProvider(_judge, RewardSource.SELF)
    out = prov.score(["t1", "t2"], [[0, 1, 2], [0, 1, 2]])
    assert out.shape == (2, 3)
    assert np.all(np.isfinite(out))


def test_judge_provider_refuses_non_finite_scores():
    bad = JudgeRewardProvider(lambda t, a: [float("nan")] * len(a))
    with pytest.raises(ValueError, match="non-finite"):
        bad.score(["t"], [[0, 1]])


def test_gold_provider_reads_l3_only_outside_a_training_scope():
    led = SupervisionLedger()
    verifier = _StubVerifier()
    prov = GoldRewardProvider(verifier, led)

    out = prov.score(["t1"], [[0, 1, 2]])
    assert out.shape == (1, 3)
    assert led.count(level=Level.L3, granted=True) == 1

    with led.training_scope(), pytest.raises(SupervisionViolation):
        prov.score(["t1"], [[0, 1, 2]])


class _StubVerifier:
    def verify(self, task_id, answers, purpose=""):
        return [1.0 if a == 0 else 0.0 for a in answers]


def test_build_arms_derives_degenerate_zeroing_from_the_source():
    """Arm (iv) must differ from arm (i) by construction.  In this module the
    config is derived, which is the fix for the protocol-A defect where the
    fourth arm silently duplicated the first."""
    led = SupervisionLedger()
    base = AdvantageConfig(eps=1e-4, degenerate_zero=False)
    stacks = build_arms(
        judge_current=_judge, judge_protected=_judge,
        gold_verifier=_StubVerifier(), ledger=led, adv_cfg=base,
        arms=("self", "degenerate_zeroed"),
    )
    self_arm = stacks[RewardSource.SELF].adv_cfg
    dz_arm = stacks[RewardSource.DEGENERATE_ZEROED].adv_cfg
    assert self_arm.degenerate_zero is False
    assert dz_arm.degenerate_zero is True
    assert dz_arm.degenerate_source == self_arm.degenerate_source


def test_build_arms_gives_gold_its_own_provider():
    led = SupervisionLedger()
    stacks = build_arms(
        judge_current=_judge, judge_protected=_judge,
        gold_verifier=_StubVerifier(), ledger=led,
        adv_cfg=AdvantageConfig(), arms=("self", "gold"),
    )
    assert isinstance(stacks[RewardSource.GOLD].provider, GoldRewardProvider)
    assert isinstance(stacks[RewardSource.SELF].provider, JudgeRewardProvider)
    # the gold stack carries the diagnostic role, not a training one
    assert stacks[RewardSource.GOLD].provider.ledger is led
