"""L1/L2/L3 supervision ledger.

The scientific claim of the paper rests on one invariant: **during training, no
method talks to the L3 gold verifier**.  This module makes that invariant a
runtime error rather than a convention.

Levels (matching the paper's supervision-access ledger)
-------------------------------------------------------
L1  Stage-initial demonstrations, synthetic negatives from rule perturbation,
    self-consistency minority voting.  Always available to training.
L2  Weak signals that are computed by the model itself: self-consistency
    quorum, self-generated-input code execution, Fisher labels on P_ext.
    Available to training, but audited separately.
L3  Gold verifiers: exact solvers, hidden tests, SymPy, IFEval checkers, BFCL
    AST matching, and any upstream score produced by them.  **Banned inside a
    training scope.**  Admissible only through an explicit non-training role
    (E1 gold DID control arm, E2 diagnostic gold reward arm, post-hoc audit,
    H6 non-inferiority reporting).

The ban is implemented with a thread-local training-depth counter.  ``Trainer``
enters ``training_scope()`` around every optimisation step; any L3 access from
any thread that is inside that scope raises ``SupervisionViolation``.  Runs are
also written to an append-only JSONL audit log so the paper can report an exact
call count.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterator


class Level(IntEnum):
    """Ordered supervision levels; higher is more privileged."""

    L1 = 1
    L2 = 2
    L3 = 3

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


class Role(str):
    """Who is asking for access.  Each role declares its ceiling."""

    # --- training-time roles (ceiling L2) -----------------------------------
    METHOD_TRAIN = "method_train"          # any method's GRPO loop
    METHOD_WARMUP = "method_warmup"        # judge warm-up / replay warm-up
    FISHER_ESTIMATION = "fisher_estimation"

    # --- diagnostic / reporting roles (ceiling L3) --------------------------
    E1_GOLD_DID_CONTROL = "e1_gold_did_control"
    E2_DIAGNOSTIC_GOLD_ARM = "e2_diagnostic_gold_arm"
    AUDIT_REPORTING = "audit_reporting"
    NI_TEST = "ni_test"
    LOSS_GRADIENT_AUDIT = "loss_gradient_audit"

    @staticmethod
    def ceiling(role: str) -> Level:
        if role in _TRAINING_ROLES:
            return Level.L2
        if role in _DIAGNOSTIC_ROLES:
            return Level.L3
        raise ValueError(f"unknown supervision role {role!r}")

    @staticmethod
    def is_training(role: str) -> bool:
        return role in _TRAINING_ROLES


_TRAINING_ROLES = frozenset(
    {Role.METHOD_TRAIN, Role.METHOD_WARMUP, Role.FISHER_ESTIMATION}
)

_DIAGNOSTIC_ROLES = frozenset(
    {
        Role.E1_GOLD_DID_CONTROL,
        Role.E2_DIAGNOSTIC_GOLD_ARM,
        Role.AUDIT_REPORTING,
        Role.NI_TEST,
        Role.LOSS_GRADIENT_AUDIT,
    }
)

ROLES_REQUIRING_GOLD_DECLARATION = _DIAGNOSTIC_ROLES - {Role.LOSS_GRADIENT_AUDIT}


class SupervisionViolation(RuntimeError):
    """Raised when a component requests supervision it is not entitled to."""


@dataclass(frozen=True)
class AccessRecord:
    seq: int
    ts: float
    level: str
    role: str
    purpose: str
    in_training_scope: bool
    granted: bool
    detail: str = ""


@dataclass
class SupervisionLedger:
    """Thread-safe access gate + audit log.

    One ledger instance is shared by an entire experiment.  It is cheap to
    construct; write it to disk with ``dump`` at the end of a run.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    strict: bool = True
    records: list[AccessRecord] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _local: threading.local = field(default_factory=threading.local, repr=False)
    _seq: int = 0
    _gold_control_calls: int = 0

    # -- training scope -----------------------------------------------------
    @property
    def training_depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @property
    def in_training_scope(self) -> bool:
        return self.training_depth > 0

    @property
    def gold_control_depth(self) -> int:
        """Depth of *declared* gold-control training scopes.

        A gold-control arm (E1's DID control, E2's diagnostic gold arm, Row 10)
        is explicitly allowed to train on L3 rewards.  It is still logged and
        counted separately so the paper can report, side by side,
        ``gold_calls_in_training = 0`` for every method and
        ``gold_calls_in_gold_control > 0`` only for the declared reference arms.
        """
        return getattr(self._local, "gold_depth", 0)

    @contextmanager
    def training_scope(self, gold_control: bool = False) -> Iterator[None]:
        """Everything inside this block is a training decision.

        ``gold_control=True`` may only be set by a run that is registered as a
        reference/diagnostic arm; the authorisation still goes through the
        role ceiling, and every L3 read is recorded with its purpose.
        """
        if gold_control:
            self._local.gold_depth = self.gold_control_depth + 1
        self._local.depth = self.training_depth + 1
        try:
            yield
        finally:
            self._local.depth -= 1
            if gold_control:
                self._local.gold_depth -= 1


    # -- access -------------------------------------------------------------
    def check(self, level: Level, role: str, purpose: str = "") -> bool:
        """Return True if allowed, False if not (never raises)."""
        try:
            self._authorize(level, role, purpose)
        except SupervisionViolation:
            return False
        return True

    def access(self, level: Level, role: str, purpose: str = "") -> "_Grant":
        """Return a grant object; raises immediately if the access is illegal."""
        self._authorize(level, role, purpose)
        return _Grant(self, level, role, purpose)

    # -- internals ----------------------------------------------------------
    def _authorize(self, level: Level, role: str, purpose: str) -> None:
        level = Level(level)
        ceiling = Role.ceiling(role)
        exempt = self.gold_control_depth > 0 and role in ROLES_REQUIRING_GOLD_DECLARATION
        reason = ""
        if level > ceiling:
            reason = (
                f"role {role!r} (ceiling {ceiling}) requested {level}"
            )
        elif level == Level.L3 and self.in_training_scope and not exempt:
            reason = (
                f"L3 access from role {role!r} inside a training scope; "
                "training-time gold calls are forbidden by the supervision ledger"
                " (only a declared gold-control training scope may read L3)"
            )
        if reason:
            self._log(level, role, purpose, granted=False, detail=reason)
            if self.strict:
                raise SupervisionViolation(reason)
        else:
            detail = "gold-control training scope" if (exempt and level == Level.L3) else ""
            if exempt and level == Level.L3:
                with self._lock:
                    self._gold_control_calls += 1
            self._log(level, role, purpose, granted=True, detail=detail)

    def _log(self, level: Level, role: str, purpose: str, *, granted: bool,
             detail: str = "") -> None:
        with self._lock:
            self._seq += 1
            self.records.append(
                AccessRecord(
                    seq=self._seq,
                    ts=time.time(),
                    level=str(level),
                    role=role,
                    purpose=purpose,
                    in_training_scope=self.in_training_scope,
                    granted=granted,
                    detail=detail,
                )
            )

    # -- reporting ----------------------------------------------------------
    def count(self, *, level: Level | None = None, granted: bool | None = None,
              role: str | None = None) -> int:
        return sum(
            1
            for r in self.records
            if (level is None or r.level == str(level))
            and (granted is None or r.granted is granted)
            and (role is None or r.role == role)
        )

    @property
    def gold_calls_in_training(self) -> int:
        """Ordinary training-time gold reads — the headline number, must be 0.

        Denied requests are *not* calls: they were refused before any verifier
        ran, and counting them would make a correctly-blocked leak look like a
        successful one.
        """
        return sum(
            1 for r in self.records
            if r.level == "L3" and r.in_training_scope and r.granted
            and r.detail != "gold-control training scope"
        )

    @property
    def gold_calls_in_gold_control(self) -> int:
        """L3 reads made by *declared* gold-control arms (Row 10, E1/E2 gold)."""
        return self._gold_control_calls

    @property
    def violations(self) -> list[AccessRecord]:
        return [r for r in self.records if not r.granted]

    def summary(self) -> dict[str, Any]:
        per_level = {
            lvl.name: self.count(level=lvl, granted=True) for lvl in Level
        }
        per_role: dict[str, int] = {}
        for r in self.records:
            if r.granted:
                per_role[r.role] = per_role.get(r.role, 0) + 1
        return {
            "run_id": self.run_id,
            "gold_calls_in_training": self.gold_calls_in_training,
            "gold_calls_in_gold_control": self.gold_calls_in_gold_control,
            "violations": len(self.violations),
            "allowed_by_level": per_level,
            "allowed_by_role": per_role,
            "total_requests": len(self.records),
        }

    def assert_clean(self, *, allow_gold_control: bool = False) -> None:
        """Fail if the run leaked gold into a trained method.

        A gold-control arm is allowed to have ``gold_calls_in_gold_control > 0``
        but must still have zero ordinary training-time gold reads;
        ``gold_calls_in_training`` already excludes the exempted records, so the
        flag is documentation of intent rather than a filter.
        """
        leaked = self.gold_calls_in_training
        if leaked:
            raise SupervisionViolation(
                f"{leaked} L3 accesses occurred inside a training scope "
                f"outside a declared gold-control arm (run {self.run_id})"
            )
        if not allow_gold_control and self.gold_calls_in_gold_control:
            raise SupervisionViolation(
                f"{self.gold_calls_in_gold_control} L3 reads came from a "
                f"gold-control arm, which this run did not declare "
                f"(run {self.run_id})"
            )

    def dump(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in self.records:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        return path


class _Grant:
    """Non-reentrant marker proving an authorised read."""

    __slots__ = ("ledger", "level", "role", "purpose")

    def __init__(self, ledger: SupervisionLedger, level: Level, role: str,
                 purpose: str) -> None:
        self.ledger = ledger
        self.level = level
        self.role = role
        self.purpose = purpose

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Grant {self.level} role={self.role} purpose={self.purpose!r}>"

    # Grants are re-entrant no-ops inside a ``with`` block: authorisation has
    # already been decided (and logged) when the grant was created, so entering
    # and leaving must not add noise to the audit trail.
    def __enter__(self) -> "_Grant":
        return self

    def __exit__(self, *exc) -> bool:
        return False
