"""Backend protocols and a fail-loud conformance check.

`Runner` used to construct `ToyModel` directly, so a real backend could only
be wired by editing the training loop.  That also meant a missing method
surfaced as an `AttributeError` on the *first training step* — after the
warm-up, the ledger, and (on the cluster) several minutes of model loading.

Both backends must satisfy the protocols below.  They are checked once, at
construction, so an incomplete backend fails immediately and names exactly
what it is missing.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, Sequence, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# task world: prompts, gold, vocabulary
# ---------------------------------------------------------------------------


@runtime_checkable
class World(Protocol):
    """The task structure a training run reads from.

    Deliberately small: these are the only world facts the training loop and
    the baselines touch.  A real benchmark backend implements them over its
    own dataset instead of over a synthetic scoring map.
    """

    vocab: int
    prompt_len: int

    def make_prompts(self, stage: Any, n: int, *, seed: int, offset: int = 0
                     ) -> tuple[torch.Tensor, list[str]]:
        ...

    def correct_token(self, stage: Any, prompt_ids: torch.Tensor) -> torch.Tensor:
        """The verifier's answer. Only ``gold``/``gold_score_matrix`` may use it,
        and those run under an L3 grant outside any training scope."""
        ...

    def demo_answer(self, stage: Any, prompt_ids: torch.Tensor) -> torch.Tensor:
        """The answer attached to a demonstration (L1). Must not call a verifier."""
        ...

    def gold(self, stage: Any, prompt_ids: torch.Tensor,
             answers: torch.Tensor) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# model: the view/adapter interface the trainer and diagnostics call
# ---------------------------------------------------------------------------


@runtime_checkable
class Model(Protocol):
    """Everything `Runner` and the experiments call on a backend."""

    # adapters
    def set_view(self, view: str) -> None: ...
    def view_parameters(self, view: str) -> list[tuple[str, torch.nn.Parameter]]: ...
    def named_parameters(self): ...
    def parameters(self): ...

    # stage + scoring
    def set_stage(self, stage: Any) -> None: ...
    def actor_logits(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor: ...
    def judge_score_matrix(self, prompt_ids: torch.Tensor, answers: torch.Tensor,
                           context_ids: torch.Tensor | None = None) -> torch.Tensor: ...
    def judge_prob(self, prompt_ids: torch.Tensor,
                   answer_ids: torch.Tensor) -> torch.Tensor: ...
    def gold_score_matrix(self, prompt_ids: torch.Tensor,
                          answers: torch.Tensor) -> torch.Tensor: ...

    # policy
    def sample(self, prompt_ids: torch.Tensor, group_size: int, **kwargs
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def logprob(self, sequences: torch.Tensor, attention_mask: torch.Tensor,
                prompt_len: int) -> torch.Tensor: ...

    # snapshot target for the frozen-judge ("protected") arm
    @property
    def judge_bank(self) -> torch.nn.Module: ...


# What training/runner.py and experiments/*.py actually call.  ``set_view`` is
# deliberately absent: it is how *one* backend switches active adapters, and
# requiring it would force every backend to adopt PEFT's model of the world.
# The portable operation is ``view_parameters(view)``, which returns the
# parameters of a role rather than mutating global state.
REQUIRED_MODEL_METHODS: tuple[str, ...] = (
    "view_parameters", "named_parameters", "parameters",
    "set_stage", "actor_logits", "judge_score_matrix", "judge_prob",
    "gold_score_matrix", "sample", "logprob",
)
REQUIRED_WORLD_METHODS: tuple[str, ...] = (
    "make_prompts", "correct_token", "demo_answer", "gold",
)


# How the training loop actually calls each method.  Presence alone is not
# enough: HFBackend.sample originally took ``Sequence[str]`` and no
# ``generator``, so it satisfied every name check while raising TypeError on
# the first training step.  Each entry is (positional args after self, kwargs).
REQUIRED_CALLS: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {
    "sample": (("prompt_ids", "group_size"), {"generator": None}),
    "logprob": (("sequences", "attention_mask", "prompt_len"), {}),
    "judge_score_matrix": (("prompt_ids", "answers"), {}),
    "judge_prob": (("prompt_ids", "answer_ids"), {}),
    "gold_score_matrix": (("prompt_ids", "answers"), {}),
    "actor_logits": (("prompt_ids",), {}),
    "view_parameters": (("actor",), {}),
    "set_stage": ((1,), {}),
}


class BackendProtocolError(TypeError):
    """A backend is missing part of the training protocol."""


def check_model(model: object, *, name: str | None = None) -> None:
    """Raise with the *complete* list of missing members, not just the first."""
    label = name or type(model).__name__
    missing = [m for m in REQUIRED_MODEL_METHODS
               if not callable(getattr(model, m, None))]
    if not hasattr(model, "judge_bank"):
        missing.append("judge_bank (property)")
    if missing:
        raise BackendProtocolError(
            f"{label} does not satisfy the model protocol; missing: "
            f"{', '.join(missing)}. Training/runner.py calls these directly, "
            "so a partial backend would fail on the first training step."
        )
    # A method can exist and still be un-callable the way the loop calls it.
    bad = _incompatible_calls(model)
    if bad:
        detail = "; ".join(f"{m}: {err}" for m, err in bad)
        raise BackendProtocolError(
            f"{label} has methods that cannot be called as training/runner.py "
            f"calls them -> {detail}"
        )


def _incompatible_calls(model: object) -> list[tuple[str, str]]:
    """Methods whose signature rejects the loop's call shape."""
    bad: list[tuple[str, str]] = []
    for name, (args, kwargs) in REQUIRED_CALLS.items():
        fn = getattr(model, name, None)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):        # builtins / C callables
            continue
        # ``fn`` came from ``getattr(instance, name)``, so it is already bound:
        # its signature has no ``self`` and passing one here would reject every
        # correctly shaped method with "too many positional arguments".
        try:
            sig.bind(*args, **kwargs)
        except TypeError as exc:
            bad.append((name, str(exc)))
    return bad


def check_world(world: object, *, name: str | None = None) -> None:
    label = name or type(world).__name__
    missing = [m for m in REQUIRED_WORLD_METHODS
               if not callable(getattr(world, m, None))]
    for attr in ("vocab", "prompt_len"):
        if not hasattr(world, attr):
            missing.append(attr)
    if missing:
        raise BackendProtocolError(
            f"{label} does not satisfy the world protocol; missing: "
            f"{', '.join(missing)}."
        )


def describe(backend: object) -> dict[str, Sequence[str] | bool]:
    """Introspection helper for `plan`/audit output: what a backend provides.

    Takes an **instance** when one is available.  A class is accepted too, but
    ``judge_bank`` is resolved per instance in some backends (the toy model
    assigns it in ``__init__``), so a class-level report cannot see it — that
    is reported as "unknown" rather than "missing", because claiming absence
    for an attribute that does exist would be a false alarm.
    """
    is_instance = not isinstance(backend, type)
    present = [m for m in REQUIRED_MODEL_METHODS
               if callable(getattr(backend, m, None))]
    absent = [m for m in REQUIRED_MODEL_METHODS if m not in present]
    bank = hasattr(backend, "judge_bank")
    return {"present": present, "absent": absent,
            "judge_bank": bank if is_instance else ("present" if bank else "unknown"),
            "complete": (not absent and bank) if is_instance else None}
