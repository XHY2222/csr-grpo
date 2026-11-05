"""The ledger is the load-bearing safety mechanism; test it adversarially."""

import pytest

from csr_grpo.supervision.ledger import (
    Level, Role, SupervisionLedger, SupervisionViolation,
)
from csr_grpo.supervision.verifiers import ToyGoldVerifier


def test_training_role_cannot_request_l3_at_all():
    led = SupervisionLedger()
    with pytest.raises(SupervisionViolation):
        led.access(Level.L3, Role.METHOD_TRAIN, "sneaky")
    assert led.count(granted=False) == 1


def test_l3_blocked_inside_training_scope_for_even_audit_role():
    led = SupervisionLedger()
    with led.training_scope():
        with pytest.raises(SupervisionViolation):
            led.access(Level.L3, Role.AUDIT_REPORTING, "inside training")
    assert led.gold_calls_in_training == 0   # denied reads are not calls
    assert len(led.violations) == 1


def test_l3_allowed_outside_training_scope():
    led = SupervisionLedger()
    v = ToyGoldVerifier(led)
    assert led.access(Level.L3, Role.AUDIT_REPORTING, "audit")
    assert led.gold_calls_in_training == 0


def test_l1_l2_allowed_inside_training():
    led = SupervisionLedger()
    with led.training_scope():
        led.access(Level.L1, Role.METHOD_WARMUP, "demo")
        led.access(Level.L2, Role.METHOD_TRAIN, "advantage")
    s = led.summary()
    assert s["violations"] == 0
    assert s["gold_calls_in_training"] == 0
    assert s["allowed_by_level"]["L1"] == 1
    assert s["allowed_by_level"]["L2"] == 1


def test_gold_control_scope_is_exempt_but_counted_separately():
    led = SupervisionLedger()
    with led.training_scope(gold_control=True):
        led.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM, "gold arm")
    s = led.summary()
    assert s["gold_calls_in_gold_control"] == 1
    assert s["gold_calls_in_training"] == 0
    led.assert_clean(allow_gold_control=True)


def test_gold_control_scope_still_rejects_training_roles():
    """Exemption is for declared diagnostic roles only, not a blanket bypass."""
    led = SupervisionLedger()
    with led.training_scope(gold_control=True):
        with pytest.raises(SupervisionViolation):
            led.access(Level.L3, Role.METHOD_TRAIN, "leak")


def test_scope_nesting_and_restoration():
    led = SupervisionLedger()
    assert not led.in_training_scope
    with led.training_scope():
        with led.training_scope():
            assert led.training_depth == 2
        assert led.training_depth == 1
    assert led.training_depth == 0


def test_scope_cleared_after_exception():
    led = SupervisionLedger()
    with pytest.raises(RuntimeError):
        with led.training_scope():
            raise RuntimeError("boom")
    assert not led.in_training_scope


def test_access_log_is_append_only_and_exportable(tmp_path):
    led = SupervisionLedger()
    led.access(Level.L1, Role.METHOD_WARMUP, "a")
    with led.training_scope():
        led.access(Level.L2, Role.METHOD_TRAIN, "b")
    seqs = [r.seq for r in led.records]
    assert seqs == sorted(seqs) and len(seqs) == 2
    out = led.dump(tmp_path / "ledger.jsonl")
    assert len(out.read_text().strip().splitlines()) == 2


def test_unknown_role_is_rejected():
    led = SupervisionLedger()
    with pytest.raises(ValueError):
        led.access(Level.L1, "made_up_role", "x")


def test_toy_verifier_routes_through_ledger():
    led = SupervisionLedger()
    v = ToyGoldVerifier(led)
    v.verify("task-1", ["t=1"], purpose="test")
    assert led.count(level=Level.L3, granted=True) == 1
    with led.training_scope(), pytest.raises(SupervisionViolation):
        v.verify("task-2", ["t=1"], purpose="should fail")


# ---------------------------------------------------------------------------
# assert_clean: both rejection paths
# ---------------------------------------------------------------------------


def test_assert_clean_rejects_gold_control_when_not_allowed():
    """``assert_clean(allow_gold_control=False)`` must refuse an undeclared gold
    arm: the run's summary can still say ``gold_calls_in_training == 0``, but
    gold-control reads are evidence of a diagnostic arm the run never declared
    -- which is exactly what this guard exists to catch."""
    from csr_grpo.supervision.ledger import SupervisionViolation

    led = SupervisionLedger()
    with led.training_scope(gold_control=True):
        led.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM, "gold arm")
    assert led.summary()["gold_calls_in_training"] == 0
    with pytest.raises(SupervisionViolation, match="gold-control arm"):
        led.assert_clean(allow_gold_control=False)


def test_assert_clean_passes_when_gold_control_is_declared_and_training_is_clean():
    led = SupervisionLedger()
    with led.training_scope(gold_control=True):
        led.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM, "gold arm")
    led.assert_clean(allow_gold_control=True)   # must not raise


def test_assert_clean_rejects_gold_in_training_even_when_gold_control_allowed():
    """A declared gold-control arm does not license *other* training-time
    gold reads; the ledger blocks the access at the source, raising
    immediately rather than recording it for ``assert_clean`` to catch."""
    from csr_grpo.supervision.ledger import SupervisionViolation

    led = SupervisionLedger()
    with led.training_scope(gold_control=True):
        led.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM, "gold arm")
    with pytest.raises(SupervisionViolation, match="training scope"):
        with led.training_scope():
            led.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM, "leak")
