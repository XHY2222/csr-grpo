"""Low-rank adapters with per-matrix coordinate masks.

One ``LowRankDelta`` is a LoRA pair ``(A, B)`` applied to a frozen base
projection::

    ΔW = B A · (alpha / r),   B initialised to zero

so the adapter starts as an exact no-op.  ``rank == 0`` means "no private (or
no shared) capacity" and yields no parameters at all — that is how the paper's
``r_S = 0`` Dual-Adapter configuration is expressed without special-casing.

Masks are applied to gradients (not weights) so Fisher-masking and stratified
random masking differ only in *which* coordinates receive gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LowRankDelta(nn.Module):
    """A single LoRA pair."""

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float | None = None, name: str = "",
                 generator: torch.Generator | None = None):
        super().__init__()
        if rank < 0:
            raise ValueError("rank must be >= 0")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / rank if rank > 0 else 0.0
        self.name = name
        if rank > 0:
            a = torch.empty(rank, in_features)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
            self.A = nn.Parameter(a)
            self.B = nn.Parameter(torch.zeros(out_features, rank))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    @property
    def is_active(self) -> bool:
        return self.rank > 0

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros(*x.shape[:-1], self.out_features,
                               dtype=x.dtype, device=x.device)
        return (x @ self.A.t()) @ self.B.t() * self.scaling

    def effective_weight(self) -> torch.Tensor:
        """``ΔW`` in the ambient (out, in) space — used for capacity proxies."""
        if not self.is_active:
            return torch.zeros(self.out_features, self.in_features)
        return (self.B @ self.A) * self.scaling

    def extra_repr(self) -> str:  # pragma: no cover
        return f"name={self.name} r={self.rank} scale={self.scaling:g}"


class AdapterBank(nn.Module):
    """Named adapters sharing one input/output space, plus coordinate masks."""

    def __init__(self, in_features: int, out_features: int,
                 specs: dict[str, tuple[int, float | None]],
                 generator: torch.Generator | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._rank = {}
        for name, (rank, alpha) in specs.items():
            self.add_module(name, LowRankDelta(in_features, out_features, rank,
                                               alpha, name=name,
                                               generator=generator))
            self._rank[name] = rank
        self.masks: dict[str, dict[str, torch.Tensor]] = {}

    # -- forward ------------------------------------------------------------
    def delta(self, x: torch.Tensor, active: Sequence[str]) -> torch.Tensor:
        out = None
        for name in active:
            mod: LowRankDelta = getattr(self, name)
            d = mod.delta(x)
            out = d if out is None else out + d
        if out is None:
            out = torch.zeros(*x.shape[:-1], self.out_features,
                              dtype=x.dtype, device=x.device)
        return out

    # -- training control ---------------------------------------------------
    def set_trainable(self, names: Iterable[str], trainable: bool = True) -> None:
        names = set(names)
        for name in self._rank:
            mod: LowRankDelta = getattr(self, name)
            if mod.A is None:
                continue
            req = trainable and name in names
            mod.A.requires_grad_(req)
            mod.B.requires_grad_(req)

    def trainable_params(self, names: Sequence[str]) -> list[tuple[str, nn.Parameter]]:
        out: list[tuple[str, nn.Parameter]] = []
        for name in names:
            mod: LowRankDelta = getattr(self, name)
            if mod.A is None:
                continue
            if mod.A.requires_grad:
                out.append((f"{name}.A", mod.A))
                out.append((f"{name}.B", mod.B))
        return out

    # -- masks --------------------------------------------------------------
    def install_mask(self, name: str, keep_A: torch.Tensor, keep_B: torch.Tensor) -> None:
        """Freeze coordinates where ``keep`` is False by zeroing their grad."""
        mod: LowRankDelta = getattr(self, name)
        if mod.A is None:
            return
        self.masks[name] = {"A": keep_A, "B": keep_B}

        def _hook(grad: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
            return grad * keep.to(grad.device, grad.dtype)

        mod.A.register_hook(lambda g, k=keep_A: _hook(g, k))
        mod.B.register_hook(lambda g, k=keep_B: _hook(g, k))

    def clear_masks(self) -> None:
        for name in list(self.masks):
            mod: LowRankDelta = getattr(self, name)
            if mod.A is not None:
                mod.A._backward_hooks.clear()
                mod.B._backward_hooks.clear()
        self.masks.clear()

    def trainable_fraction(self, name: str) -> float:
        m = self.masks.get(name)
        if m is None:
            return 1.0
        a, b = m["A"], m["B"]
        return float((a.sum() + b.sum()) / (a.numel() + b.numel()))

    # -- capacity proxies ---------------------------------------------------
    def effective_rank(self, name: str) -> float:
        mod: LowRankDelta = getattr(self, name)
        if not mod.is_active:
            return 0.0
        w = mod.effective_weight().detach()
        if w.abs().max() == 0:
            return 0.0
        s = torch.linalg.svdvals(w)
        p = s ** 2
        p = p / p.sum()
        # exponential-entropy effective rank (Roy & Vetterli)
        return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))

    def frobenius(self, name: str) -> float:
        mod: LowRankDelta = getattr(self, name)
        return float(mod.effective_weight().detach().norm())


@dataclass
class AdapterSpec:
    """Declarative description of one architecture configuration."""

    r_shared: int = 32
    r_actor_private: int = 32
    r_judge_private: int = 32
    alpha: float = 64.0
    actor_active: tuple[str, ...] = ("shared", "actor")
    judge_active: tuple[str, ...] = ("shared", "judge")

    @classmethod
    def dual_adapter(cls, r: int = 64, alpha: float = 128.0) -> "AdapterSpec":
        """``r_S = 0, r_A = r_J = r`` (paper Row 8 / Group C)."""
        return cls(r_shared=0, r_actor_private=r, r_judge_private=r,
                   alpha=alpha, actor_active=("actor",), judge_active=("judge",))

    @classmethod
    def shared(cls, r: int = 32, alpha: float = 64.0,
               r_private: int | None = None) -> "AdapterSpec":
        """Fixed mixed architecture: one shared adapter plus per-role privates.

        ``r`` is the *shared* rank; ``r_private`` is the private rank (defaults
        to ``r``).  The paper's Group D configuration is
        ``shared(r=32, r_private=32)``; ``shared(r=32, r_private=0)`` removes the
        private branch and leaves only the maskable shared one.
        """
        priv = r if r_private is None else r_private
        return cls(r_shared=r, r_actor_private=priv, r_judge_private=priv,
                   alpha=alpha, actor_active=("shared", "actor"),
                   judge_active=("shared", "judge"))

    def specs(self) -> dict[str, tuple[int, float]]:
        """Private adapters only; the shared one is a single object on the model."""
        return {
            "actor": (self.r_actor_private, self.alpha),
            "judge": (self.r_judge_private, self.alpha),
        }

    def private_rank(self) -> int:
        return self.r_actor_private + self.r_judge_private


def build_bank(spec: AdapterSpec, in_features: int, out_features: int,
               generator: torch.Generator | None = None) -> AdapterBank:
    return AdapterBank(in_features, out_features, spec.specs(), generator=generator)


def build_shared(spec: AdapterSpec, in_features: int, out_features: int,
                 generator: torch.Generator | None = None) -> LowRankDelta:
    """The single adapter object both views use when ``r_S > 0``.

    Sharing one ``nn.Module`` instance (rather than two equal-valued copies) is
    what makes cross-role gradient contamination a real property: a gradient
    step taken for the judge view changes the actor view's parameters.
    """
    return LowRankDelta(in_features, out_features, spec.r_shared,
                        alpha=spec.alpha, name="shared", generator=generator)

