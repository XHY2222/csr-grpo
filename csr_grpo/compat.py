"""Small compatibility shims.

The development machine ships a PyTorch built against NumPy 1.x next to NumPy
2.2, so ``Tensor.numpy()`` raises ``RuntimeError: Numpy is not available`` while
NumPy itself works fine.  Every array conversion in this package therefore goes
through ``to_np``, which prefers the zero-copy bridge and falls back to
``tolist`` — correctness first, and on a healthy host the fast path is used.

Training tensors here are tiny (``G=8`` rollouts), so the fallback costs
nothing; the real GPU runs happen on the cluster where the bridge works.
"""

from __future__ import annotations

import numpy as np
import torch


def to_np(x) -> np.ndarray:
    """Convert a tensor (or anything array-like) to a float64 NumPy array."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        try:
            return t.numpy().astype(np.float64)
        except RuntimeError:
            return np.asarray(t.tolist(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def to_np_any(x) -> np.ndarray:
    """Like ``to_np`` but preserves dtype (for integer answer ids)."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        try:
            return t.numpy()
        except RuntimeError:
            return np.asarray(t.tolist())
    return np.asarray(x)


def to_tensor(x, dtype=torch.float64) -> torch.Tensor:
    """NumPy → torch without going through the torch+numpy bridge."""
    arr = np.asarray(x)
    if arr.dtype == object:  # pragma: no cover - defensive
        arr = arr.astype(np.float64)
    return torch.tensor(arr.tolist(), dtype=dtype)


def numpy_bridge_ok() -> bool:
    """True when ``Tensor.numpy()`` works on this host."""
    try:
        torch.zeros(1).numpy()
        return True
    except RuntimeError:
        return False
