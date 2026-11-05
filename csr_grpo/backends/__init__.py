"""Backend registry.

The toy CPU backend is always available; the HuggingFace backend registers
itself lazily because ``transformers``/``peft`` may be absent (they are on the
dev box, but the cluster image is the supported path for real runs).
"""

from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable[..., object]] = {}


def register_backend(name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorator used by ``hf.build_hf_backend`` and friends."""

    def _wrap(fn):
        _REGISTRY[name] = fn
        return fn

    return _wrap


def available_backends() -> list[str]:
    names = {"toy"}
    if "hf" in _REGISTRY:
        names.add("hf")
    return sorted(names)


def build_backend(name: str, **kwargs):
    """Construct a backend by name; ``toy`` is wired here to avoid a cycle."""
    if name == "toy":
        from .toy import ToyModel, ToyWorld
        from ..methods.lora import AdapterSpec

        spec = kwargs.pop("spec", AdapterSpec.dual_adapter())
        model = ToyModel(ToyWorld(), spec, **kwargs)
        return model
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {available_backends()}")
    return _REGISTRY[name](**kwargs)


def __getattr__(name: str):
    # PEP 562: importing ``csr_grpo.backends.hf`` is enough to register it, but
    # touching ``backends.hf`` should also work without an explicit import.
    if name == "hf":
        import importlib

        return importlib.import_module(".hf", __name__)
    raise AttributeError(name)
