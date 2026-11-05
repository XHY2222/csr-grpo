"""Exact, restorable training state.

Protocol A of E2 requires four branches to start from *identical* Actor
parameters, *identical* rollouts and *identical* Adam moments.  Adam is
stateful (``exp_avg``, ``exp_avg_sq``, ``step``) and gradient clipping depends
on the current gradient norm, so copying only the weights is not enough — a
branch would silently inherit another branch's momentum.

``TrainingState`` snapshots:

* every named parameter tensor (device-agnostic clones)
* every optimizer entry keyed by parameter name (all state dicts, not only
  ``exp_avg``/``exp_avg_sq``, so e.g. AMSGrad's ``max_exp_avg_sq`` survives)
* torch / numpy / python RNG states
* a global step counter

and restores them atomically.  ``assert_equal`` is used by tests to prove the
rollback is bit-exact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import torch


def _clone_param(p: torch.Tensor) -> torch.Tensor:
    return p.detach().clone().to("cpu")


def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: Mapping) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass
class TrainingState:
    """A bit-exact snapshot of parameters + optimizer + RNG."""

    params: dict[str, torch.Tensor] = field(default_factory=dict)
    opt_state: dict[str, dict] = field(default_factory=dict)
    opt_defaults: dict = field(default_factory=dict)
    rng: dict = field(default_factory=dict)
    step: int = 0
    tags: dict[str, str] = field(default_factory=dict)

    # -- capture ------------------------------------------------------------
    @classmethod
    def capture(cls, module: torch.nn.Module, optimizer: torch.optim.Optimizer,
                step: int = 0, **tags: str) -> "TrainingState":
        params = {name: _clone_param(p) for name, p in module.named_parameters()}
        by_id = {id(p): name for name, p in module.named_parameters()}
        opt_state: dict[str, dict] = {}
        for p, st in optimizer.state.items():
            name = by_id.get(id(p))
            if name is None:
                raise KeyError("optimizer holds a parameter not owned by module")
            opt_state[name] = {
                k: (_clone_param(v) if isinstance(v, torch.Tensor) else v)
                for k, v in st.items()
            }
        return cls(params=params, opt_state=opt_state,
                   opt_defaults=dict(optimizer.defaults),
                   rng=_rng_state(), step=step, tags=dict(tags))

    # -- restore ------------------------------------------------------------
    def restore(self, module: torch.nn.Module, optimizer: torch.optim.Optimizer,
                *, restore_rng: bool = True) -> None:
        with torch.no_grad():
            for name, p in module.named_parameters():
                if name not in self.params:
                    raise KeyError(f"snapshot missing parameter {name!r}")
                p.copy_(self.params[name].to(p.device, dtype=p.dtype))
        optimizer.state.clear()
        name_to_param = dict(module.named_parameters())
        for name, st in self.opt_state.items():
            p = name_to_param.get(name)
            if p is None:
                raise KeyError(f"snapshot refers to unknown parameter {name!r}")
            optimizer.state[p] = {
                k: (v.clone().to(p.device) if isinstance(v, torch.Tensor) else v)
                for k, v in st.items()
            }
        if restore_rng:
            _set_rng_state(self.rng)

    # -- comparisons --------------------------------------------------------
    def max_param_diff(self, module: torch.nn.Module) -> float:
        diffs = []
        with torch.no_grad():
            for name, p in module.named_parameters():
                ref = self.params[name].to(p.device, dtype=p.dtype)
                diffs.append(float((p - ref).abs().max()))
        return max(diffs) if diffs else 0.0

    def param_delta(self, module: torch.nn.Module) -> dict[str, torch.Tensor]:
        """``θ_current − θ_snapshot`` per parameter (the Δθ of protocol A)."""
        deps = {}
        with torch.no_grad():
            for name, p in module.named_parameters():
                ref = self.params[name].to(p.device, dtype=p.dtype)
                deps[name] = (p - ref).detach().clone()
        return deps

    def state_diff(self, other: "TrainingState") -> dict[str, float]:
        """Largest absolute difference in params and each Adam buffer."""
        out = {"param": _max_abs(self.params, other.params)}
        keys = set(self.opt_state) | set(other.opt_state)
        for k in keys:
            a = self.opt_state.get(k, {})
            b = other.opt_state.get(k, {})
            for sk in set(a) | set(b):
                va, vb = a.get(sk), b.get(sk)
                if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                    out[f"{k}.{sk}"] = float((va - vb).abs().max())
                elif va != vb:
                    out[f"{k}.{sk}"] = float("inf")
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"TrainingState(step={self.step}, params={len(self.params)}, tags={self.tags})"


def _max_abs(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    if set(a) != set(b):
        return float("inf")
    if not a:
        return 0.0
    return max(float((a[k] - b[k]).abs().max()) for k in a)


@dataclass
class ModuleState:
    """Parameter-only snapshot of a submodule (e.g. the judge adapter bank).

    Used for the frozen "protected judge" arm: switching views must not touch
    the optimizer, so this is deliberately separate from ``TrainingState``.
    """

    params: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def capture(cls, module: torch.nn.Module) -> "ModuleState":
        return cls(params={n: _clone_param(p) for n, p in module.named_parameters()})

    def restore(self, module: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, p in module.named_parameters():
                p.copy_(self.params[name].to(p.device, dtype=p.dtype))

    def context(self, module: torch.nn.Module):
        return _ModuleSwap(module, self)


class _ModuleSwap:
    def __init__(self, module: torch.nn.Module, state: ModuleState):
        self.module = module
        self.state = state
        self.live = ModuleState.capture(module)

    def __enter__(self):
        self.state.restore(self.module)
        return self

    def __exit__(self, *exc):
        self.live.restore(self.module)
        return False


# ---------------------------------------------------------------------------
# rollout buffers: identical rollouts across branches
# ---------------------------------------------------------------------------


@dataclass
class RolloutBatch:
    """Frozen on-policy rollouts shared by all protocol-A branches."""

    task_ids: list[str]
    sequences: torch.Tensor            # (P, G, T)
    attention_mask: torch.Tensor       # (P, G, T)
    old_logprob: torch.Tensor          # (P*G, T-1) log probs at rollout time
    prompt_len: int
    sampled_at_step: int
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.sequences.shape[0]), int(self.sequences.shape[1])

    def collapse(self) -> "RolloutBatch":
        """(P, G, ...) → (P*G, ...) for a single vectorised forward pass.

        With ``P == 1`` the leading axis is dropped rather than merged, because
        a batch of one prompt *is* already the collapsed form; merging would
        produce a 1-D tensor and silently lose the time axis.  A batch that is
        already collapsed (2-D) is returned unchanged.
        """
        if self.sequences.ndim == 2:
            return self
        p, g = self.shape
        seqs = self.sequences.reshape(p * g, *self.sequences.shape[2:])
        mask = self.attention_mask.reshape(p * g, *self.attention_mask.shape[2:])
        return RolloutBatch(
            task_ids=[t for t in self.task_ids for _ in range(g)],
            sequences=seqs,
            attention_mask=mask,
            old_logprob=self.old_logprob,
            prompt_len=self.prompt_len,
            sampled_at_step=self.sampled_at_step,
            meta=dict(self.meta),
        )

    def clone(self) -> "RolloutBatch":
        return RolloutBatch(
            task_ids=list(self.task_ids),
            sequences=self.sequences.clone(),
            attention_mask=self.attention_mask.clone(),
            old_logprob=self.old_logprob.clone(),
            prompt_len=self.prompt_len,
            sampled_at_step=self.sampled_at_step,
            meta=dict(self.meta),
        )

    def fingerprint(self) -> str:
        import hashlib

        from ..compat import to_np_any

        h = hashlib.sha256()
        h.update(np.ascontiguousarray(to_np_any(self.sequences)).tobytes())
        h.update(np.ascontiguousarray(to_np_any(self.old_logprob)).tobytes())
        return h.hexdigest()[:16]
