"""Gold verifiers (L3) — every call is gated by the supervision ledger.

A verifier object holds a reference to the experiment ledger.  ``__call__``
requests an L3 grant for a named role; if the caller is inside a training scope
the ledger raises ``SupervisionViolation`` before the verifier does any work.
That is the single chokepoint the whole codebase relies on, so nothing else is
allowed to compute gold labels.

The deterministic toy verifier is used by the CPU testbed; the real ones are
thin adapters over external solvers and are imported lazily so the package
stays installable without them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, Sequence, TypeVar

from .ledger import Level, Role, SupervisionLedger

T = TypeVar("T")


class GoldVerifier(Protocol):
    """Callable returning per-answer gold correctness in {0.0, 1.0}."""

    name: str

    def verify(self, task_id: str, answers: Sequence[str]) -> list[float]: ...


class _GatedVerifier:
    """Mixin that funnels every gold read through the ledger."""

    name: str = "gold"

    def __init__(self, ledger: SupervisionLedger, role: str = Role.AUDIT_REPORTING):
        self._ledger = ledger
        self._role = role

    def verify(self, task_id: str, answers: Sequence[str],
               purpose: str = "") -> list[float]:
        self._ledger.access(Level.L3, self._role, purpose or f"{self.name}:{task_id}")
        return self._verify_impl(task_id, answers)

    def _verify_impl(self, task_id: str, answers: Sequence[str]) -> list[float]:
        raise NotImplementedError

    # convenience: exactly one answer
    def verify_one(self, task_id: str, answer: str, purpose: str = "") -> float:
        return self.verify(task_id, [answer], purpose)[0]


def _stable_unit(*parts: str) -> float:
    """Deterministic value in [0, 1) from string parts (toy gold labels)."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass
class ToyGoldVerifier(_GatedVerifier):
    ledger: SupervisionLedger = None  # type: ignore[assignment]

    """Deterministic pseudo-gold for the CPU testbed.

    A ``(task_id, answer)`` pair is correct iff the answer's committed token is
    the single designated token for that task.  This reproduces the essential
    structure of a real verifier — a threshold on exact match — while staying
    fast and dependency-free.
    """

    name: str = "toy_gold"
    n_choices: int = 4
    accept_threshold: float = 0.5

    def __post_init__(self) -> None:
        # dataclass bypasses _GatedVerifier.__init__; wire the ledger explicitly
        if self.ledger is None:
            raise ValueError("ToyGoldVerifier requires a SupervisionLedger")
        self._ledger = self.ledger
        self._role = getattr(self, "role", Role.AUDIT_REPORTING)

    def correct_token(self, task_id: str) -> int:
        return int(_stable_unit("token", task_id) * self.n_choices)

    def token_score(self, task_id: str, answer: str) -> float:
        # answers are strings like "t=2" emitted by the toy policy
        try:
            tok = int(answer.split("=")[-1])
        except ValueError:
            return 0.0
        return 1.0 if tok == self.correct_token(task_id) else 0.0

    def _verify_impl(self, task_id: str, answers: Sequence[str]) -> list[float]:
        return [self.token_score(task_id, a) for a in answers]


# ---------------------------------------------------------------------------
# Real verifiers (lazy, not exercised on the dev machine)
# ---------------------------------------------------------------------------


class ExactSolverVerifier(_GatedVerifier):
    """Stage 1 logic: exact truth-table / constraint solver."""

    name = "exact_solver"

    def __init__(self, ledger: SupervisionLedger, solver: Callable[[str, str], bool],
                 role: str = Role.AUDIT_REPORTING):
        super().__init__(ledger, role)
        self._solver = solver

    def _verify_impl(self, task_id: str, answers: Sequence[str]) -> list[float]:
        return [1.0 if self._solver(task_id, a) else 0.0 for a in answers]


class HiddenTestVerifier(_GatedVerifier):
    """Stage 2 code: run generated code against hidden unit tests in a sandbox."""

    name = "hidden_tests"

    def __init__(self, ledger: SupervisionLedger, runner: Callable[[str, str], bool],
                 role: str = Role.AUDIT_REPORTING):
        super().__init__(ledger, role)
        self._runner = runner

    def _verify_impl(self, task_id: str, answers: Sequence[str]) -> list[float]:
        return [1.0 if self._runner(task_id, a) else 0.0 for a in answers]


class SymPyVerifier(_GatedVerifier):
    """Stage 3 math: symbolic equivalence of boxed answers."""

    name = "sympy"

    def __init__(self, ledger: SupervisionLedger, role: str = Role.AUDIT_REPORTING):
        super().__init__(ledger, role)

    def _verify_impl(self, task_id: str, answers: Sequence[str]) -> list[float]:
        import sympy  # noqa: F401  (imported here so the package is optional)

        from .reference_answers import REFERENCE  # type: ignore

        ref = sympy.sympify(REFERENCE[task_id])
        out = []
        for a in answers:
            try:
                out.append(1.0 if sympy.simplify(sympy.sympify(a) - ref) == 0 else 0.0)
            except Exception:
                out.append(0.0)
        return out


class ConstraintCheckerVerifier(_GatedVerifier):
    """Stage 4 IFEval / BFCL: deterministic constraint script + AST match.

    Disabled during training by construction — the checker *is* gold.
    """

    name = "ifbench_checker"

    def __init__(self, ledger: SupervisionLedger,
                 checker: Callable[[str, str], bool],
                 role: str = Role.AUDIT_REPORTING):
        super().__init__(ledger, role)
        self._checker = checker

    def _verify_impl(self, task_id: str, answers: Sequence[str]) -> list[float]:
        return [1.0 if self._checker(task_id, a) else 0.0 for a in answers]


def make_verifier(stage: str, ledger: SupervisionLedger, **kw) -> GoldVerifier:
    """Factory used by run scripts; keeps stage→verifier wiring in one place."""
    stage = stage.lower()
    if stage in {"toy", "0"}:
        return ToyGoldVerifier(ledger, **kw)
    if stage in {"logic", "stage1"}:
        return ExactSolverVerifier(ledger, **kw)
    if stage in {"code", "stage2"}:
        return HiddenTestVerifier(ledger, **kw)
    if stage in {"math", "stage3"}:
        return SymPyVerifier(ledger, **kw)
    if stage in {"if", "stage4"}:
        return ConstraintCheckerVerifier(ledger, **kw)
    raise ValueError(f"unknown stage {stage!r}")
