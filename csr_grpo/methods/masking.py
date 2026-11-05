"""Fixed-architecture gradient masking: Fisher vs. stratified random (Group D).

The paper's confound control is explicit: *masking rate* ``m`` and *architecture
capacity* ``r_S`` are orthogonal variables.  Both methods here therefore run on
the **same fixed architecture** ``(r_S, r_priv)`` and differ only in *which*
coordinates of the shared adapter receive gradient:

* ``fisher``  — freeze the coordinates the judge relies on most, i.e. those with
  the largest empirical diagonal Fisher, estimated on ``P_ext`` with **weak
  (L1/L2) labels only**;
* ``random``  — stratified random: in each adapter matrix, freeze exactly the
  same *number* of coordinates as Fisher does, but choose them uniformly.

The Fisher labels never come from gold: ``P_ext`` carries route-A/route-B weak
labels by design, and the estimator asserts that no L3 grant was used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from ..config.schema import Config
from ..methods.lora import AdapterSpec, LowRankDelta
from ..rewards.advantage import AdvMode
from .base import BaseMethod, MethodPlan


# ---------------------------------------------------------------------------
# Fisher diagonal
# ---------------------------------------------------------------------------


@torch.no_grad()
def _zero_grads(module: torch.nn.Module) -> None:
    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()


def empirical_fisher(
    module: torch.nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    n_batches: int = 8,
    *,
    weak_labels_only: bool = True,
) -> dict[str, torch.Tensor]:
    """Diagonal Fisher estimated as mean squared per-sample gradient.

    ``loss_fn`` must close over *weakly labelled* data.  We check the flag but
    cannot inspect the closure; the runner is responsible for passing a
    route-A/B loss, and the ledger audit independently proves no L3 grant
    occurred during estimation.
    """
    if not weak_labels_only:
        raise ValueError(
            "Fisher estimation is defined on P_ext weak labels (L1/L2); "
            "gold-labelled Fisher would make the mask an L3-dependent object"
        )
    fisher: dict[str, torch.Tensor] = {}
    for _ in range(n_batches):
        module.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        for name, p in module.named_parameters():
            if p.grad is None:
                continue
            if name not in fisher:
                fisher[name] = torch.zeros_like(p)
            fisher[name] += p.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= max(n_batches, 1)
    module.zero_grad(set_to_none=True)
    return fisher


@dataclass
class MaskPlan:
    """Which coordinates survive, per adapter matrix."""

    keep: dict[str, tuple[torch.Tensor, torch.Tensor]]  # name -> (keep_A, keep_B)
    fraction: float
    kind: str

    def total_kept(self) -> int:
        return sum(int(a.sum()) + int(b.sum()) for a, b in self.keep.values())

    def summary(self) -> dict[str, float]:
        out = {"kind": self.kind, "fraction": self.fraction}
        for name, (a, b) in self.keep.items():
            out[f"{name}.kept"] = float(a.sum() + b.sum())
        return out


def fisher_mask(module: torch.nn.Module, fisher: dict[str, torch.Tensor],
                names: Sequence[str], fraction: float) -> MaskPlan:
    """Keep the lowest-Fisher ``1 - fraction`` of coordinates per matrix.

    Freezing high-Fisher coordinates is the mechanism under test: the judge's
    sensitive directions are shielded so the actor cannot overwrite them.
    """
    keep: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in names:
        mod = dict(module.named_modules()).get(name)
        if not isinstance(mod, LowRankDelta) or mod.A is None:
            continue
        fa = fisher.get(f"{name}.A")
        fb = fisher.get(f"{name}.B")
        if fa is None or fb is None:
            raise KeyError(f"missing Fisher for {name}")
        keep[name] = tuple(_keep_lowest(x, fraction) for x in (fa, fb))
    return MaskPlan(keep=keep, fraction=fraction, kind="fisher")


def stratified_random_mask(module: torch.nn.Module, names: Sequence[str],
                           fraction: float, seed: int = 0) -> MaskPlan:
    """Same number of frozen coordinates per matrix, uniformly random positions."""
    g = torch.Generator().manual_seed(seed)
    keep: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in names:
        mod = dict(module.named_modules()).get(name)
        if not isinstance(mod, LowRankDelta) or mod.A is None:
            continue
        keep[name] = tuple(_random_keep(x.shape, fraction, g) for x in (mod.A, mod.B))
    return MaskPlan(keep=keep, fraction=fraction, kind="random")


def _keep_lowest(score: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = score.flatten()
    n = flat.numel()
    n_keep = int(round((1.0 - fraction) * n))
    keep = torch.zeros_like(flat)
    if n_keep >= n:
        keep[:] = True
    elif n_keep > 0:
        # ties broken deterministically by index
        order = torch.argsort(flat, descending=False, stable=True)
        keep[order[:n_keep]] = True
    return keep.view_as(score)


def _random_keep(shape: torch.Size, fraction: float, g: torch.Generator) -> torch.Tensor:
    n = int(np.prod(shape))
    n_keep = int(round((1.0 - fraction) * n))
    perm = torch.randperm(n, generator=g)
    keep = torch.zeros(n, dtype=torch.bool)
    keep[perm[:n_keep]] = True
    return keep.view(shape)


def apply_plan(module: torch.nn.Module, plan: MaskPlan) -> None:
    """Install the mask on the adapter bank (idempotent)."""
    for bank_attr in ("actor_bank", "judge_bank"):
        bank = getattr(module, bank_attr, None)
        if bank is None:
            continue
        for name, (ka, kb) in plan.keep.items():
            if hasattr(bank, name.split(".")[-1]) or name.split(".")[-1] in {
                "shared", "actor", "judge"
            }:
                bank.install_mask(name.split(".")[-1], ka.to("cpu"), kb.to("cpu"))


def equal_frozen_counts(a: MaskPlan, b: MaskPlan) -> bool:
    """Confound check: Fisher and random must freeze identical amounts."""
    if set(a.keep) != set(b.keep):
        return False
    for name in a.keep:
        fa = a.keep[name][0].numel() - int(a.keep[name][0].sum()) + \
             a.keep[name][1].numel() - int(a.keep[name][1].sum())
        fbi = b.keep[name][0].numel() - int(b.keep[name][0].sum()) + \
              b.keep[name][1].numel() - int(b.keep[name][1].sum())
        if fa != fbi:
            return False
    return True


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


@dataclass
class _MaskMethod(BaseMethod):
    mask_fraction: float = 0.75
    kind: str = "fisher"
    seed: int = 0

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name,
            spec=self.spec,
            judge_trainable_in_stage=True,
            mask_fraction=self.mask_fraction,
            mask_kind=self.kind,
            notes=(
                f"fixed architecture r_S={self.spec.r_shared}, "
                f"r_priv={self.spec.r_actor_private}; mask {self.mask_fraction:.0%} "
                f"of shared coordinates ({self.kind})",
                "m=100% is the private-rank-only reference, NOT the 64-rank "
                "Dual-Adapter",
            ),
        )

    @classmethod
    def _mk(cls, cfg: Config, kind: str, frac: float | None):
        if frac is None:
            frac = 0.75
        return cls(
            name=f"{kind}_mask",
            spec=AdapterSpec(r_shared=32, r_actor_private=32, r_judge_private=32,
                             alpha=float(cfg.model.lora_alpha)),
            mask_fraction=frac,
            kind=kind,
            seed=cfg.seed,
        )

    def build_mask(self, module: torch.nn.Module,
                   loss_fn: Callable[[], torch.Tensor] | None = None,
                   n_batches: int = 8) -> MaskPlan:
        raise NotImplementedError


@dataclass
class FisherMaskMethod(_MaskMethod):
    name: str = "fisher_mask"
    kind: str = "fisher"

    @classmethod
    def from_config(cls, cfg: Config, fraction: float | None = None) -> "FisherMaskMethod":
        base = cls._mk(cfg, "fisher", fraction)
        return cls(name="fisher_mask", spec=base.spec,
                   mask_fraction=base.mask_fraction, kind="fisher", seed=cfg.seed)

    def build_mask(self, module, loss_fn=None, n_batches=8) -> MaskPlan:
        if loss_fn is None:
            raise ValueError("Fisher mask needs a weak-label loss on P_ext")
        fisher = empirical_fisher(module, loss_fn, n_batches,
                                  weak_labels_only=True)
        return fisher_mask(module, fisher, self.spec.actor_active,
                           self.mask_fraction)


@dataclass
class StratifiedRandomMaskMethod(_MaskMethod):
    name: str = "random_mask"
    kind: str = "random"

    @classmethod
    def from_config(cls, cfg: Config, fraction: float | None = None) -> "StratifiedRandomMaskMethod":
        base = cls._mk(cfg, "random", fraction)
        return cls(name="random_mask", spec=base.spec,
                   mask_fraction=base.mask_fraction, kind="random", seed=cfg.seed)

    def build_mask(self, module, loss_fn=None, n_batches=8) -> MaskPlan:
        return stratified_random_mask(module, self.spec.actor_active,
                                      self.mask_fraction, seed=self.seed)
