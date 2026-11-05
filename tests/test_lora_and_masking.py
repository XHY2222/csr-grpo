"""Adapters start as exact no-ops, masking perturbs gradients only, Fisher is L1/L2."""

import numpy as np
import pytest
import torch

from csr_grpo.backends.toy import ToyModel, ToyWorld
from csr_grpo.methods.lora import AdapterBank, AdapterSpec, LowRankDelta
from csr_grpo.methods.masking import (
    equal_frozen_counts, empirical_fisher, fisher_mask, stratified_random_mask,
)


def test_lora_starts_as_identity():
    m = LowRankDelta(6, 5, rank=4)
    x = torch.randn(3, 6)
    assert torch.allclose(m.delta(x), torch.zeros(3, 5))


def test_rank_zero_has_no_parameters():
    m = LowRankDelta(6, 5, rank=0)
    assert m.A is None and list(m.parameters()) == []
    assert torch.allclose(m.delta(torch.randn(2, 6)), torch.zeros(2, 5))


def test_effective_weight_shape():
    m = LowRankDelta(6, 5, rank=3, alpha=6.0)
    assert m.effective_weight().shape == (5, 6)
    assert m.scaling == pytest.approx(2.0)


def test_dual_adapter_has_no_shared_parameters():
    spec = AdapterSpec.dual_adapter(r=8)
    model = ToyModel(ToyWorld(), spec, seed=0)
    assert model.shared.A is None
    actor_names = {n for n, _ in model.view_parameters("actor")}
    judge_names = {n for n, _ in model.view_parameters("judge")}
    assert actor_names.isdisjoint(judge_names)


def test_shared_architecture_really_shares():
    """r_S > 0 must expose the SAME tensors to both views (contamination path)."""
    spec = AdapterSpec.shared(r=8)
    model = ToyModel(ToyWorld(), spec, seed=0)
    actor = dict(model.view_parameters("actor"))
    judge = dict(model.view_parameters("judge"))
    assert "shared.A" in actor and "shared.A" in judge
    assert actor["shared.A"] is judge["shared.A"]


def test_shared_gradient_reaches_view_parameters():
    """The judge loss must flow into `shared`, which the actor view also reads.

    With ``B`` zero-initialised, ``∂L/∂A = Bᵀ(…) = 0`` while
    ``∂L/∂B = (x Aᵀ)ᵀ(…) != 0``; the shared object is the same tensor in both
    views, so a judge-side step really does move the actor's parameters.
    """
    spec = AdapterSpec.shared(r=8)
    model = ToyModel(ToyWorld(), spec, seed=0)
    model.set_stage(1)
    pid = torch.zeros(2, 4, dtype=torch.long)
    ans = torch.tensor([1, 2])
    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model.judge_logits(pid, ans), ans).backward()
    assert model.shared.B.grad is not None and model.shared.B.grad.abs().sum() > 0
    # and the two views resolve to the very same tensor
    assert dict(model.view_parameters('actor'))['shared.B'] is dict(
        model.view_parameters('judge'))['shared.B']


def test_mask_freezes_exact_fraction():
    spec = AdapterSpec(r_shared=0, r_actor_private=8, r_judge_private=8, alpha=16.0)
    model = ToyModel(ToyWorld(), spec, seed=0)
    plan = stratified_random_mask(model, ["actor"], fraction=0.75, seed=0)
    kept = plan.total_kept()
    total = sum(a.numel() + b.numel() for a, b in plan.keep.values())
    assert kept == pytest.approx(0.25 * total, abs=1)


def test_fisher_and_random_freeze_equal_counts():
    spec = AdapterSpec(r_shared=0, r_actor_private=8, r_judge_private=8, alpha=16.0)
    model = ToyModel(ToyWorld(), spec, seed=0)
    model.set_stage(1)
    pid = torch.zeros(4, 4, dtype=torch.long)
    ans = torch.tensor([1, 2, 3, 0])

    def loss_fn():
        return torch.nn.functional.cross_entropy(model.judge_logits(pid, ans), ans)

    fisher = empirical_fisher(model, loss_fn, n_batches=3)
    fp = fisher_mask(model, fisher, ["actor"], fraction=0.75)
    rp = stratified_random_mask(model, ["actor"], fraction=0.75, seed=1)
    assert equal_frozen_counts(fp, rp)


def test_fisher_requires_weak_labels():
    model = ToyModel(ToyWorld(), AdapterSpec.dual_adapter(r=4), seed=0)
    with pytest.raises(ValueError):
        empirical_fisher(model, lambda: torch.tensor(0.0), weak_labels_only=False)


def test_fisher_keeps_low_fisher_coordinates():
    from csr_grpo.methods.masking import _keep_lowest
    score = torch.tensor([1.0, 4.0, 2.0, 3.0])
    keep = _keep_lowest(score, fraction=0.5)
    # freeze the two largest (4.0, 3.0) -> keep indices 0 and 2
    assert keep.tolist() == [True, False, True, False]
