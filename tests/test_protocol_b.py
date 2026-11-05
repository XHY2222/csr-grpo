"""E2 protocol B: independent on-policy resampling per branch.

Protocol A freezes one rollout; protocol B lets each branch re-sample from its
own updated policy every step, measuring the closed-loop cumulative effect.
These tests pin the properties that distinguish the two.
"""

import numpy as np
import pytest
import torch

from csr_grpo.config.schema import Config
from csr_grpo.experiments.e2_intervention import (
    BranchResult, paired_branch_differences, run_protocol_b,
)
from csr_grpo.supervision.ledger import SupervisionLedger


def _cfg(lr=1e-2):
    cfg = Config()
    cfg.method = "dual_adapter"
    cfg.model.r_actor = 16
    cfg.model.lora_alpha = 32
    cfg.train.group_size = 4
    cfg.train.warmup_rounds = 4
    cfg.train.lr = lr
    cfg.data.audit_prompts = 12
    cfg.data.n_prompts_per_stage = 24
    return cfg


def test_protocol_b_returns_all_four_arms():
    led = SupervisionLedger()
    out = run_protocol_b(_cfg(), led, stage=2, audit_stage=1, k_steps=3, n_eval=12)
    assert set(out) == {"self", "protected", "gold", "degenerate_zeroed"}
    for arm, br in out.items():
        assert len(br.acc_old) == len(br.acc_new) >= 1, arm


def test_protocol_b_keeps_training_gold_at_zero():
    led = SupervisionLedger()
    run_protocol_b(_cfg(), led, stage=2, audit_stage=1, k_steps=3, n_eval=8)
    s = led.summary()
    assert s["gold_calls_in_training"] == 0
    assert s["violations"] == 0
    assert s["gold_calls_in_gold_control"] > 0


def test_protocol_b_branches_actually_diverge():
    """The point of B is closed-loop divergence; identical branches mean it is
    not re-sampling independently (or the step size is unobservably small).

    Note: the toy backend often gives all-zero ΔAcc at small step counts
    because greedy argmax on a 35-50% accuracy model with a tiny LoRA moves
    the predicted token only when the adapter pushes one logit past another.
    The test therefore checks *at least one* contrast is non-zero rather than
    every contrast: a single arm whose policy drift moved even one prompt's
    argmax is sufficient evidence of independent divergence.
    """
    out = run_protocol_b(_cfg(lr=1e-2), SupervisionLedger(), stage=2,
                         audit_stage=1, k_steps=6, n_eval=12)
    contrasts = paired_branch_differences(out)
    assert set(contrasts) == {"self", "protected", "degenerate_zeroed"}
    for arm, c in contrasts.items():
        assert "d_acc_old_vs_gold" in c and "d_acc_new_vs_gold" in c
    # at least one arm must produce a non-zero contrast; a zero everywhere
    # means something prevented any of the branches from moving
    any_nonzero = any(
        abs(c["d_acc_old_vs_gold"]) > 1e-9 or abs(c["d_acc_new_vs_gold"]) > 1e-9
        for c in contrasts.values()
    )
    assert any_nonzero, (
        "protocol B produced identical ΔAcc for every arm — branches are not "
        "diverging; check that each arm re-samples from its own updated policy "
        "and that the optimizer step actually changes the actor parameters"
    )


def test_branch_result_deltas_are_last_minus_first():
    br = BranchResult(arm="x", acc_old=[0.5, 0.4, 0.3], acc_new=[0.2, 0.2],
                      acc_old_start=0.6, acc_new_start=0.1)
    assert br.d_acc_old == pytest.approx(0.3 - 0.6)
    assert br.d_acc_new == pytest.approx(0.2 - 0.1)
    assert br.as_dict()["arm"] == "x"


def test_paired_branch_differences_exclude_reference():
    # d_acc = last - start, so give both branches the same start to isolate
    # the contrast arithmetic
    a = BranchResult("gold", [0.5], [0.5], 0.5, 0.5)      # dOld = 0.0
    b = BranchResult("self", [0.4], [0.4], 0.5, 0.5)      # dOld = -0.1
    out = {"gold": a, "self": b, "protected": b}
    c = paired_branch_differences(out)
    assert "gold" not in c            # the reference has no contrast against itself
    assert c["self"]["d_acc_old_vs_gold"] == pytest.approx(-0.1)
    assert c["protected"]["d_acc_old_vs_gold"] == pytest.approx(-0.1)


def test_protocol_b_uses_pass1_not_pass_at_g():
    """`max over group` would saturate near 1.0; pass@1 must not."""
    import inspect

    from csr_grpo.experiments import e2_intervention

    src = inspect.getsource(e2_intervention.run_protocol_b)
    assert "argmax" in src
    assert "max(axis=1)" not in src
