"""2x2 decomposition: the identity must hold exactly, and drift must vanish for a frozen judge."""

import numpy as np
import pytest

from csr_grpo.diagnostics.factorial import (
    SnapshotPair, boundary_attribution, factorial_attribution,
)
from csr_grpo.diagnostics.fidelity import advantages


def _pair(seed=0, n=40, k=4, drift=0.0, shift=0.0):
    rng = np.random.default_rng(seed)
    gold0 = (rng.random((n, k)) > 0.5).astype(float)
    goldk = (rng.random((n, k)) > 0.5).astype(float)
    j_start_0 = gold0 * 0.8 + rng.random((n, k)) * 0.2
    j_end_0 = j_start_0 + drift * rng.normal(size=(n, k))
    # ``shift`` must move *judge quality* on Y_k, so scale gold confidence down
    # (adding a constant would leave every judge ranking unchanged).
    j_start_k = goldk * 0.8 * (1.0 - shift) + rng.random((n, k)) * 0.2
    j_end_k = j_start_k + drift * rng.normal(size=(n, k))
    return SnapshotPair(
        judge_start_on_y0=j_start_0, judge_end_on_y0=j_end_0,
        judge_start_on_yk=j_start_k, judge_end_on_yk=j_end_k,
        gold_on_y0=gold0, gold_on_yk=goldk,
    )


def test_identity_holds_exactly():
    res = factorial_attribution(_pair(), n_resamples=50)
    assert res.interaction_identity_residual < 1e-9


def test_frozen_judge_gives_zero_drift_term():
    """Dual-Adapter: judge is frozen inside the stage, so E10 == E00 exactly."""
    p = _pair(drift=0.0)
    p.judge_end_on_y0 = p.judge_start_on_y0.copy()
    p.judge_end_on_yk = p.judge_start_on_yk.copy()
    res = factorial_attribution(p, n_resamples=50)
    assert abs(res.terms["judge_drift(ΔJ=E10-E00)"].point) < 1e-12


def test_total_equals_sum_of_parts():
    res = factorial_attribution(_pair(drift=0.05, shift=0.1), n_resamples=50)
    t = res.terms
    parts = (t["judge_drift(ΔJ=E10-E00)"].point
             + t["output_shift(ΔY=E01-E00)"].point
             + t["interaction(I_int)"].point)
    assert parts == pytest.approx(t["total(E11-E00)"].point, abs=1e-9)


def test_output_shift_captured_when_answers_degrade():
    """Y_k scored worse by the *same* judge must show up in ΔY, not ΔJ."""
    rng = np.random.default_rng(9)
    n, k = 30, 4
    gold = (rng.random((n, k)) > 0.5).astype(float)
    good = gold * 0.9 + rng.random((n, k)) * 0.1
    bad = gold * 0.1 + rng.random((n, k)) * 0.2
    p = SnapshotPair(judge_start_on_y0=good, judge_end_on_y0=good,
                     judge_start_on_yk=bad, judge_end_on_yk=bad,
                     gold_on_y0=gold, gold_on_yk=gold)
    res = factorial_attribution(p, n_resamples=50, kind="acc_rank")
    assert res.terms["judge_drift(ΔJ=E10-E00)"].point == pytest.approx(0.0, abs=1e-12)
    assert res.terms["output_shift(ΔY=E01-E00)"].point > 0


def test_boundary_term_detects_warmup_degradation():
    rng = np.random.default_rng(11)
    n = 30
    gold = (rng.random((n, 4)) > 0.5).astype(float)
    good = gold * 0.9 + rng.random((n, 4)) * 0.1
    worse = (1.0 - gold) * 0.9 + rng.random((n, 4)) * 0.1   # ranks inverted
    chain = boundary_attribution(good, worse, gold, _pair(), n_resamples=50)
    # E_w = E(after) - E(before) > 0 means warm-up hurt discrimination
    assert chain.E_w.point > 0
    assert "E_w(warmup boundary)" in chain.flatten()


def test_shape_mismatch_rejected():
    p = _pair()
    p.gold_on_y0 = p.gold_on_y0[:-1]
    with pytest.raises(ValueError):
        p.cells()


def test_measure_factorial_without_canon_scores_omits_canon_key():
    """When canon_scores is not supplied the 'canon' key must be absent."""
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    cfg.model.r_actor = 16; cfg.train.group_size = 4
    cfg.data.audit_prompts = 8; cfg.train.warmup_rounds = 3

    runner = Runner(cfg, make_method(cfg).plan(), SupervisionLedger())
    ctx = runner.make_context(2, n_prompts=8, seed=0, audit_stage=1,
                              audit_seed=5001)
    y0 = runner.capture_stage_start(ctx, audit_stage=1)
    out = runner.measure_factorial(ctx, 1, y0, audit_stage=1)
    assert "terms" in out
    assert "canon" not in out


def test_measure_factorial_with_canon_scores_includes_delta_adv_canon():
    """When canon_scores is provided the output carries a 'canon' sub-dict
    with 'delta_adv_canon', wiring the ΔÂ_canon probe into the factorial."""
    import numpy as np
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    cfg.model.r_actor = 16; cfg.train.group_size = 4
    cfg.data.audit_prompts = 8; cfg.train.warmup_rounds = 3

    runner = Runner(cfg, make_method(cfg).plan(), SupervisionLedger())
    ctx = runner.make_context(2, n_prompts=8, seed=0, audit_stage=1,
                              audit_seed=5001)
    y0 = runner.capture_stage_start(ctx, audit_stage=1)

    rng = np.random.default_rng(42)
    P, G = len(ctx.audit_prompts), cfg.train.group_size
    orig = rng.random((P, G))
    rew = orig + rng.normal(0, 0.05, (P, G))   # small rewrite perturbation

    out = runner.measure_factorial(ctx, 1, y0, audit_stage=1,
                                   canon_scores=(orig, rew))
    assert "canon" in out
    assert "delta_adv_canon" in out["canon"]
    assert isinstance(out["canon"]["delta_adv_canon"], float)


def test_measure_factorial_second_stage_emits_boundary_chain_and_h6():
    """After two capture_stage_start calls, measure_factorial must include
    'boundary_chain' (E_w) and 'h6_non_inferiority' in its output.
    On the first stage these keys must be absent (no prev_stage_judge yet).
    """
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    cfg.model.r_actor = 16; cfg.train.group_size = 4
    cfg.data.audit_prompts = 8; cfg.train.warmup_rounds = 3

    runner = Runner(cfg, make_method(cfg).plan(), SupervisionLedger())

    # Stage 1→2 transition: first capture sets protected_judge, no prev yet
    ctx1 = runner.make_context(2, n_prompts=8, seed=0, audit_stage=1,
                               audit_seed=5001)
    y0_s1 = runner.capture_stage_start(ctx1, audit_stage=1)
    out1 = runner.measure_factorial(ctx1, 1, y0_s1, audit_stage=1)
    assert "boundary_chain" not in out1, \
        "first stage must not emit boundary_chain (no prev judge)"
    assert "h6_non_inferiority" not in out1

    # Stage 2→3 transition: second capture sets prev_stage_judge = stage-1 end
    ctx2 = runner.make_context(3, n_prompts=8, seed=100, audit_stage=2,
                               audit_seed=5002)
    y0_s2 = runner.capture_stage_start(ctx2, audit_stage=2)
    out2 = runner.measure_factorial(ctx2, 1, y0_s2, audit_stage=2)
    assert "boundary_chain" in out2, \
        "second stage must emit boundary_chain for E_w (H6)"
    assert "h6_non_inferiority" in out2
    ni = out2["h6_non_inferiority"]
    assert "non_inferior" in ni
    assert "margin" in ni
    assert abs(ni["margin"] - cfg.exp.delta_ni_pt / 100.0) < 1e-9
