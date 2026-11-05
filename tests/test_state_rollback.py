"""State capture/restore must be bit-exact, including Adam moments and RNG."""

import numpy as np
import pytest
import torch

from csr_grpo.methods.lora import AdapterSpec
from csr_grpo.training.state import ModuleState, RolloutBatch, TrainingState
from csr_grpo.backends.toy import ToyModel, ToyWorld


def _model():
    return ToyModel(ToyWorld(), AdapterSpec.dual_adapter(r=8), seed=3)


def _opt(model):
    return torch.optim.Adam(model.view_parameters("actor") and
                            [p for _, p in model.view_parameters("actor")], lr=1e-2)


def _step(model, opt, scale=1.0):
    opt.zero_grad(set_to_none=True)
    loss = sum((p ** 2).sum() for _, p in model.view_parameters("actor")) * scale
    loss.backward()
    opt.step()


def test_restore_returns_exact_parameters():
    m = _model(); opt = _opt(m)
    for _ in range(3):
        _step(m, opt)
    snap = TrainingState.capture(m, opt, step=3)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for _ in range(4):
        _step(m, opt, scale=2.0)
    assert snap.max_param_diff(m) > 0
    snap.restore(m, opt)
    for n, p in m.named_parameters():
        assert torch.equal(p.detach(), before[n]), n


def test_restore_recovers_adam_moments():
    m = _model(); opt = _opt(m)
    for _ in range(3):
        _step(m, opt)
    snap = TrainingState.capture(m, opt)
    # snapshot *values*: Adam increments `step` in place, so a live reference
    # would silently change while the test runs
    moments_before = [
        (s["exp_avg"].clone(), s["exp_avg_sq"].clone(),
         float(torch.as_tensor(s["step"]).clone()))
        for s in opt.state.values()
    ]
    for _ in range(3):
        _step(m, opt, scale=3.0)
    snap.restore(m, opt)
    moments_after = [
        (s["exp_avg"], s["exp_avg_sq"], float(torch.as_tensor(s["step"])))
        for s in opt.state.values()
    ]
    for (a1, a2, a3), (b1, b2, b3) in zip(moments_before, moments_after):
        assert torch.equal(a1, b1) and torch.equal(a2, b2)
        assert a3 == b3


def test_two_branches_from_same_state_diverge_identically():
    """Protocol A's premise: same state + same gradient => identical Δθ."""
    m = _model(); opt = _opt(m)
    for _ in range(2):
        _step(m, opt)
    snap = TrainingState.capture(m, opt)

    snap.restore(m, opt); _step(m, opt, 1.5)
    d1 = snap.param_delta(m)

    snap.restore(m, opt); _step(m, opt, 1.5)
    d2 = snap.param_delta(m)

    for k in d1:
        assert torch.allclose(d1[k], d2[k], atol=0, rtol=0), k


def test_rng_state_restored():
    m = _model(); opt = _opt(m)
    snap = TrainingState.capture(m, opt)
    a = torch.randn(5)
    snap.restore(m, opt)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_state_diff_reports_zero_after_restore():
    m = _model(); opt = _opt(m)
    for _ in range(2):
        _step(m, opt)
    snap = TrainingState.capture(m, opt)
    again = TrainingState.capture(m, opt)
    diff = snap.state_diff(again)
    assert max(diff.values()) == 0.0


def test_module_state_swap_roundtrip():
    m = _model()
    ms = ModuleState.capture(m.judge_bank)
    live_before = {n: p.clone() for n, p in m.judge_bank.named_parameters()}
    with ms.context(m.judge_bank):
        for p in m.judge_bank.parameters():
            p.data.add_(1.0)
        assert all(float((p - live_before[n]).abs().max()) == 1.0
                   for n, p in m.judge_bank.named_parameters())
    for n, p in m.judge_bank.named_parameters():
        assert torch.equal(p.data, live_before[n])


def test_rollout_batch_collapse_is_idempotent():
    b = RolloutBatch(task_ids=["a"], sequences=torch.zeros(1, 4, 6),
                     attention_mask=torch.ones(1, 4, 6),
                     old_logprob=torch.zeros(4, 5), prompt_len=5,
                     sampled_at_step=0)
    c = b.collapse()
    assert c.sequences.shape == (4, 6)
    assert c.collapse() is c


def test_fingerprint_is_order_sensitive():
    a = RolloutBatch(["x"], torch.zeros(1, 2, 3), torch.ones(1, 2, 3),
                     torch.zeros(2, 2), 2, 0)
    b = RolloutBatch(["x"], torch.zeros(1, 2, 3) + 1, torch.ones(1, 2, 3),
                     torch.zeros(2, 2), 2, 0)
    assert a.fingerprint() != b.fingerprint()
