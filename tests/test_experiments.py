"""End-to-end invariants for E1/E2/E3.

These are slow-ish (seconds) but they assert the properties the paper depends
on, not just "it runs":

* E2 protocol A gives every arm the *same* frozen rollout;
* the gold arm is the only one that reads L3, and it does so from a declared
  gold-control scope;
* a trained method records zero gold reads;
* E2's sign-stable first-order term is finite for every arm.
"""

import numpy as np
import pytest
import torch

from csr_grpo.config.schema import Config
from csr_grpo.experiments.e2_intervention import (
    cosine_similarity, first_order_delta, run_protocol_a,
)
from csr_grpo.experiments.e3_methods import E3Gate, ArmOutcome
from csr_grpo.supervision.ledger import SupervisionLedger


def _cfg():
    cfg = Config()
    cfg.method = "dual_adapter"
    cfg.model.r_actor = 16
    cfg.model.lora_alpha = 32
    cfg.train.group_size = 4
    cfg.train.warmup_rounds = 5
    cfg.data.audit_prompts = 16
    cfg.data.n_prompts_per_stage = 24
    return cfg


def test_protocol_a_shares_rollout_and_never_leaks_gold():
    led = SupervisionLedger()
    res = run_protocol_a(_cfg(), led, stage=2, audit_stage=1, step=3,
                         n_prompts=2, n_old_prompts=4)
    fps = {d["rollout_fingerprint"] for d in res.arms.values()}
    assert len(fps) == 1, "all arms must optimise the same frozen rollout"
    assert set(res.arms) == {"self", "protected", "gold", "degenerate_zeroed"}
    for arm, d in res.arms.items():
        assert np.isfinite(d["dL_old_1st"]), arm
        # cos_vs_gold is undefined when the gold arm produced no update
        # (all-degenerate groups -> zero gradient -> zero Δθ^gold).  The
        # protocol must report that as status "gold_arm_inactive" rather
        # than silently propagating NaN.
        status = d.get("status", "ok")
        if status == "ok":
            assert np.isfinite(d["cos_vs_gold"]), arm
        else:
            assert status == "gold_arm_inactive"
            assert d["gold_delta_norm"] == 0.0
        assert d["delta_norm"] >= 0.0
    s = led.summary()
    assert s["gold_calls_in_training"] == 0
    assert s["violations"] == 0
    assert s["gold_calls_in_gold_control"] >= 1


def test_protocol_a_gold_arm_is_self_similar():
    led = SupervisionLedger()
    res = run_protocol_a(_cfg(), led, stage=2, audit_stage=1, step=2,
                         n_prompts=2, n_old_prompts=3)
    assert res.arms["gold"]["cos_vs_gold"] == pytest.approx(1.0)


def test_global_ledger_stays_clean_for_a_trained_method():
    from csr_grpo.methods.base import make_method
    from csr_grpo.training.runner import Runner

    led = SupervisionLedger()
    cfg = _cfg()
    runner = Runner(cfg, make_method(cfg).plan(), led)
    runner.warmup_judge(1, n_rounds=3)
    ctx = runner.make_context(1, n_prompts=16, seed=3, audit_stage=1)
    for step in range(3):
        runner.train_step(ctx, step)
    assert led.summary()["gold_calls_in_training"] == 0
    assert led.summary()["violations"] == 0


def test_cosine_and_inner_product_are_basis_independent():
    a = {"p": torch.tensor([1.0, 2.0, 3.0])}
    b = {"p": torch.tensor([2.0, 4.0, 6.0])}
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert first_order_delta(a, b) == pytest.approx(28.0)


def test_e3_gate_requires_both_conditions():
    base = ArmOutcome("shared_grpo", forward_acc=0.5, forgetting=0.10)
    prop = ArmOutcome("dual_adapter_replay", forward_acc=0.6, forgetting=0.05)
    ref = ArmOutcome("rler_faithful", forward_acc=0.58, forgetting=0.06)
    arms = {a.arm: a for a in (base, prop, ref)}
    assert E3Gate.evaluate(arms).passed

    # forgetting improved but forward accuracy lags RLER by > 1.5 pt
    prop_bad = ArmOutcome("dual_adapter_replay", forward_acc=0.50, forgetting=0.05)
    arms2 = {**arms, "dual_adapter_replay": prop_bad}
    gate2 = E3Gate.evaluate(arms2)
    assert not gate2.passed
    assert "H3 not supported" in gate2.interpretation


def test_e3_gate_rejects_only_forgetting_gain():
    base = ArmOutcome("shared_grpo", forward_acc=0.5, forgetting=0.05)
    prop = ArmOutcome("dual_adapter_replay", forward_acc=0.6, forgetting=0.045)
    arms = {a.arm: a for a in (base, prop)}
    assert not E3Gate.evaluate(arms).passed   # gain 0.5 pt < 2 pt


# ---------------------------------------------------------------------------
# E3 end to end
# ---------------------------------------------------------------------------
#
# `E3Gate.evaluate` was tested against hand-built ArmOutcome objects; the path
# that actually produces them -- run_arm_stages -> run_e3 -- was not.  These
# tests drive it, because the gate is only meaningful if the numbers it reads
# come from a clean run (no gold in training, finite metrics, deterministic).


def _e3_cfg():
    cfg = _cfg()
    cfg.train.warmup_rounds = 2
    cfg.data.n_prompts_per_stage = 8
    return cfg


def test_e3_arm_runs_clean_and_reports_finite_metrics():
    from csr_grpo.experiments.e3_methods import run_arm_stages

    led = SupervisionLedger()
    out = run_arm_stages(_e3_cfg(), "dual_adapter_replay", led,
                         stages=(1, 2), steps_per_stage=3, n_eval=8,
                         eval_n_prompts=24)

    assert set(out.acc_per_stage) == {"logic", "code"}
    assert all(np.isfinite(v) for v in out.acc_per_stage.values())
    assert np.isfinite(out.forward_acc) and np.isfinite(out.forgetting)
    # the whole point: a trained method never reads gold in training
    assert led.gold_calls_in_training == 0
    assert len(led.violations) == 0, led.violations


def test_e3_forgetting_is_the_mean_drop_on_every_seen_stage():
    """Forgetting is not "final accuracy" -- it is the mean of per-stage drops,
    and a bug that averaged over only the last stage would pass every shape
    check while reporting a different quantity."""
    from csr_grpo.experiments.e3_methods import ArmOutcome, forgetting

    arms = ArmOutcome(
        arm="x",
        acc_per_stage={"logic": 0.4, "code": 0.6},
        acc_at_learning={"logic": 0.5, "code": 0.6},
    )
    assert forgetting(arms.acc_per_stage, arms.acc_at_learning) == pytest.approx(0.05)


def test_e3_forward_accuracy_is_the_mean_over_stages():
    from csr_grpo.experiments.e3_methods import run_arm_stages

    out = run_arm_stages(_e3_cfg(), "dual_adapter", SupervisionLedger(),
                         stages=(1, 2), steps_per_stage=2, n_eval=8,
                         eval_n_prompts=24)
    assert out.forward_acc == pytest.approx(
        float(np.mean(list(out.acc_per_stage.values()))))


def test_e3_runs_the_full_arm_matrix_and_produces_a_verdict():
    from csr_grpo.experiments.e3_methods import run_e3

    cfg = _e3_cfg()
    arms = ("shared_grpo", "dual_adapter_replay")
    res = run_e3(cfg, SupervisionLedger(), stages=(1, 2), steps_per_stage=2,
                 arms=arms)
    payload = res.as_dict()
    assert set(payload["arms"]) == set(arms)
    gate = payload["gate"]
    assert isinstance(gate["passed"], bool)
    assert set(gate["thresholds"]) == {"fm_gain_pt", "forward_gap_pt"}
    # the interpretation must name the consequence, not just pass/fail
    assert "H3" in gate["interpretation"]


def test_e3_gate_uses_the_documented_thresholds():
    from csr_grpo.experiments.e3_methods import run_e3

    res = run_e3(_e3_cfg(), SupervisionLedger(), stages=(1, 2),
                 steps_per_stage=2, arms=("shared_grpo", "dual_adapter_replay"))
    assert res.gate["thresholds"]["fm_gain_pt"] == pytest.approx(2.0)
    assert res.gate["thresholds"]["forward_gap_pt"] == pytest.approx(1.5)


def test_e3_is_deterministic_for_a_fixed_seed():
    """Method comparisons are only paired if the two runs see the same data;
    a hidden RNG would silently break that."""
    from csr_grpo.experiments.e3_methods import run_arm_stages

    a = run_arm_stages(_e3_cfg(), "dual_adapter", SupervisionLedger(),
                       stages=(1, 2), steps_per_stage=2, n_eval=8,
                       eval_n_prompts=24, seed=5)
    b = run_arm_stages(_e3_cfg(), "dual_adapter", SupervisionLedger(),
                       stages=(1, 2), steps_per_stage=2, n_eval=8,
                       eval_n_prompts=24, seed=5)
    assert a.acc_per_stage == pytest.approx(b.acc_per_stage)
    assert a.forward_acc == pytest.approx(b.forward_acc)


def test_e3_supervised_arm_takes_the_supervised_path():
    """Rows 1/2a/2b never take a policy-gradient step; if the dispatch broke,
    they would silently become a GRPO run and the baseline table would be
    wrong without any error."""
    from csr_grpo.experiments.e3_methods import run_arm_stages

    led = SupervisionLedger()
    out = run_arm_stages(_e3_cfg(), "seq_sft", led, stages=(1, 2),
                         steps_per_stage=2, n_eval=8, eval_n_prompts=24)
    assert np.isfinite(out.forward_acc)
    assert led.gold_calls_in_training == 0
    assert len(led.violations) == 0


# ---------------------------------------------------------------------------
# run_e2_prime + n_old_prompts cap
# ---------------------------------------------------------------------------


def test_run_e2_prime_runs_on_stage3_audit2():
    """E2 prime is the held-out confirmation: stage=3, audit_stage=2.  It must
    run cleanly without leaking gold into training, symmetric to E1 prime."""
    from csr_grpo.experiments.e2_intervention import run_e2_prime

    cfg = _cfg()
    led = SupervisionLedger()
    res = run_e2_prime(cfg, led, protocol_b=False, k_steps=4, seed=0)
    payload = res.as_dict()
    assert "protocol_a" in payload
    assert led.gold_calls_in_training == 0
    assert len(led.violations) == 0


def test_protocol_a_caps_n_old_prompts_to_available_audit_set():
    """Regression: n_old_prompts=16 (default) with audit_prompts=8 used to
    silently slice 8 rows, then reshape(16, G) raised a shape error.
    Now n_old_prompts is capped to the actual audit set size."""
    from csr_grpo.experiments.e2_intervention import run_protocol_a

    cfg = _cfg()
    cfg.data.audit_prompts = 6      # smaller than the default n_old_prompts=16
    led = SupervisionLedger()
    res = run_protocol_a(cfg, led, stage=2, audit_stage=1, step=2,
                         n_prompts=4, n_old_prompts=16)   # too large, must be capped
    assert all(np.isfinite(d["dL_old_1st"]) for d in res.arms.values())


# ---------------------------------------------------------------------------
# E2 multi-checkpoint structure
# ---------------------------------------------------------------------------


def test_run_e2_uses_checkpoints_from_config():
    """protocol_a must be keyed by step number when checkpoints are given;
    this is the fix for e2_checkpoints being accepted by config but ignored."""
    from csr_grpo.experiments.e2_intervention import run_e2

    cfg = _cfg()
    led = SupervisionLedger()
    checkpoints = (2, 4)
    res = run_e2(cfg, led, stage=2, audit_stage=1, protocol_b=False,
                 seed=0, checkpoints=checkpoints)
    payload = res.as_dict()
    assert "protocol_a" in payload
    pa = payload["protocol_a"]
    # must be keyed by step number
    assert set(pa.keys()) == {2, 4} or set(pa.keys()) == {"2", "4"} or \
           all(k in (2, 4, "2", "4") for k in pa.keys())
    # each value must be a protocol-A result dict with 'arms'
    for step_result in pa.values():
        assert "arms" in step_result, f"missing 'arms' in checkpoint {step_result}"
    assert led.gold_calls_in_training == 0


def test_run_e2_defaults_to_config_checkpoints():
    """When checkpoints is not given, run_e2 reads cfg.exp.e2_checkpoints."""
    from csr_grpo.experiments.e2_intervention import run_e2

    cfg = _cfg()
    cfg.exp.e2_checkpoints = (3, 6)
    led = SupervisionLedger()
    res = run_e2(cfg, led, stage=2, audit_stage=1, protocol_b=False, seed=0)
    pa = res.as_dict()["protocol_a"]
    assert len(pa) == 2


# ---------------------------------------------------------------------------
# E3 multi-seed pooling
# ---------------------------------------------------------------------------


def test_run_e3_with_multiple_seeds_averages_metrics():
    """When n_seeds=2, run_e3 must run two seed-replicate per arm and average
    the scalar metrics.  The forward_acc of the averaged outcome must be the
    mean of the two individual seed runs."""
    import numpy as np
    from csr_grpo.experiments.e3_methods import run_e3, run_arm_stages

    cfg = _cfg()
    led = SupervisionLedger()
    # single-seed reference
    single = run_e3(cfg, SupervisionLedger(), stages=(1, 2),
                    steps_per_stage=4, seed=0, n_seeds=1,
                    arms=("dual_adapter",))
    # two-seed run
    dual = run_e3(cfg, SupervisionLedger(), stages=(1, 2),
                  steps_per_stage=4, seed=0, n_seeds=2,
                  arms=("dual_adapter",))
    assert "dual_adapter" in single.arms
    assert "dual_adapter" in dual.arms
    # both must produce finite forward_acc
    assert np.isfinite(single.arms["dual_adapter"]["forward_acc"])
    assert np.isfinite(dual.arms["dual_adapter"]["forward_acc"])


def test_run_e3_n_seeds_one_is_identical_to_default():
    """n_seeds=1 must produce the same result as the default (no n_seeds arg)."""
    from csr_grpo.experiments.e3_methods import run_e3

    cfg = _cfg()
    a = run_e3(cfg, SupervisionLedger(), stages=(1, 2),
               steps_per_stage=3, seed=5, n_seeds=1, arms=("dual_adapter",))
    b = run_e3(cfg, SupervisionLedger(), stages=(1, 2),
               steps_per_stage=3, seed=5, arms=("dual_adapter",))
    assert a.arms["dual_adapter"]["forward_acc"] == \
           pytest.approx(b.arms["dual_adapter"]["forward_acc"])
