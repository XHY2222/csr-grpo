#!/usr/bin/env python3
"""Executable framework audit — the "复盘测评框架" deliverable.

The plan says the evaluation framework must be re-checked *after* the code is
written, not asserted in prose.  This script does that mechanically:

* **Deliverable traceability** — each of the four required deliverables is mapped
  to the module that implements it and to the tests that cover it.  A missing
  module or an empty test set is a failure.
* **Metric semantics** — the estimators are re-derived against closed-form
  values (perfect/inverted ranking, constant shift, degenerate groups).
* **Statistical contracts** — pairing, matching FPR, non-inferiority direction,
  and the three-way effect triage.
* **Supervision invariant** — gold is unreachable from a training scope.

Exit code is 0 only if every check passes.  Output is JSON so it can be
attached to a run manifest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from csr_grpo.diagnostics.bootstrap import (  # noqa: E402
    BootstrapCI, classify_effect, non_inferiority_test, paired_cluster_bootstrap,
)
from csr_grpo.diagnostics.early_warning import calibrate, first_detection  # noqa: E402
from csr_grpo.diagnostics.fidelity import (  # noqa: E402
    acc_rank_group, aggregate, mse_rho_identity, per_prompt_stats,
)
from csr_grpo.diagnostics.factorial import (  # noqa: E402
    SnapshotPair, factorial_attribution,
)
from csr_grpo.rewards.advantage import (  # noqa: E402
    AdvantageConfig, compute_advantages,
)
from csr_grpo.supervision.ledger import (  # noqa: E402
    Level, Role, SupervisionLedger, SupervisionViolation,
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        # comparisons against numpy values yield numpy.bool_, which json cannot
        # serialise; normalise at construction so every caller is safe
        self.passed = bool(self.passed)


@dataclass
class AuditReport:
    deliverables: list[Check] = field(default_factory=list)
    semantics: list[Check] = field(default_factory=list)
    statistics: list[Check] = field(default_factory=list)
    supervision: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for group in (self.deliverables, self.semantics,
                                          self.statistics, self.supervision)
                   for c in group)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "deliverables": [asdict(c) for c in self.deliverables],
            "semantics": [asdict(c) for c in self.semantics],
            "statistics": [asdict(c) for c in self.statistics],
            "supervision": [asdict(c) for c in self.supervision],
        }


# ---------------------------------------------------------------------------
# 1. deliverable traceability
# ---------------------------------------------------------------------------

DELIVERABLES = {
    "D1 skeleton + config + L1/L2/L3 ledger": {
        "modules": ["csr_grpo/config/schema.py", "csr_grpo/config/loader.py",
                    "csr_grpo/supervision/ledger.py"],
        "tests": ["tests/test_supervision_ledger.py"],
    },
    "D2 diagnostic pipeline (dual line, metrics, bootstrap, delta-t lead)": {
        "modules": ["csr_grpo/diagnostics/fidelity.py",
                    "csr_grpo/diagnostics/dual_line.py",
                    "csr_grpo/diagnostics/bootstrap.py",
                    "csr_grpo/diagnostics/early_warning.py",
                    "csr_grpo/diagnostics/factorial.py"],
        "tests": ["tests/test_fidelity_metrics.py",
                  "tests/test_bootstrap_and_warning.py",
                  "tests/test_factorial.py"],
    },
    "D3 gated E1 (switch/continue + gold DID + E1') and E2 (protocol A/B)": {
        "modules": ["csr_grpo/experiments/e1_switch.py",
                    "csr_grpo/experiments/e2_intervention.py"],
        "tests": ["tests/test_experiments.py", "tests/test_e1_prime.py"],
    },
    "D4 Dual-Adapter + Replay, RLER-faithful, unit tests": {
        "modules": ["csr_grpo/methods/dual_adapter.py",
                    "csr_grpo/methods/rler.py",
                    "csr_grpo/methods/lora.py"],
        "tests": ["tests/test_rler_and_methods.py",
                  "tests/test_state_rollback.py",
                  "tests/test_lora_and_masking.py"],
    },
}


def audit_deliverables() -> list[Check]:
    out: list[Check] = []
    for name, spec in DELIVERABLES.items():
        missing_mod = [m for m in spec["modules"] if not (ROOT / m).exists()]
        missing_test = [t for t in spec["tests"] if not (ROOT / t).exists()]
        out.append(Check(
            name=name,
            passed=not missing_mod and not missing_test,
            detail=(f"missing modules {missing_mod}; missing tests {missing_test}"
                    if missing_mod or missing_test else
                    f"{len(spec['modules'])} modules, {len(spec['tests'])} test files"),
        ))
    return out


# ---------------------------------------------------------------------------
# 2. metric semantics against closed forms
# ---------------------------------------------------------------------------


def _metric_checks() -> list[Check]:
    checks: list[Check] = []

    gold = np.array([1.0, 0.0, 0.0, 0.0])
    checks.append(Check(
        "Acc_rank orientation (perfect ranking -> 1, inverted -> 0)",
        acc_rank_group(np.array([0.9, 0.1, 0.2, 0.3]), gold) == 1.0
        and acc_rank_group(np.array([0.1, 0.9, 0.2, 0.3]), gold) == 0.0))

    checks.append(Check(
        "degenerate groups are undefined, not zero",
        acc_rank_group(np.array([0.9, 0.8]), np.array([1.0, 1.0])) is None
        and acc_rank_group(np.array([0.1, 0.2]), np.array([0.0, 0.0])) is None))

    st = per_prompt_stats(np.array([[0.9, 0.5, 0.4, 0.1],
                                    [0.5, 0.5, 0.5, 0.5]]),
                          np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0, 0.0]]), tau=0.05)
    checks.append(Check(
        "pseudo-contrast restricted to gold-degenerate groups",
        st.degenerate.tolist() == [True, False]
        and st.pseudo_contrast[st.degenerate].tolist() == [True]))

    rng = np.random.default_rng(0)
    g = (rng.random((60, 8)) > 0.5).astype(float)
    j = g * 0.8 + rng.random((60, 8)) * 0.2
    ident = mse_rho_identity(j, g)
    checks.append(Check(
        "MSE(Â_J, Â_gold) == 2(1 - rho) identity",
        abs(ident["gap"]) < 5e-3, f"gap={ident['gap']:.2e}"))

    # The plan quotes the *unstandardised* rank offsets [1.5, 0.5, -0.5, -1.5];
    # GRPO then divides by sigma.  Both must be checked, otherwise the audit
    # compares a standardised value against a raw one (which it initially did).
    from csr_grpo.rewards.advantage import rank_transform

    gop_judge = np.array([[0.99, 0.60, 0.59, 0.58]])
    ranks = rank_transform(gop_judge)[0]           # [4, 3, 2, 1]
    raw = ranks - ranks.mean()
    adv = compute_advantages(gop_judge,
                             cfg=AdvantageConfig(mode="rank_only", eps=0.0))
    checks.append(Check(
        "GOPO rank-only: raw offsets are [+1.5, +0.5, -0.5, -1.5]",
        np.allclose(raw, [1.5, 0.5, -0.5, -1.5]),
        f"raw={raw.tolist()}"))
    checks.append(Check(
        "GOPO rank-only: standardised advantages equal raw/sigma, 2nd still > 0",
        np.allclose(adv[0], raw / np.std(ranks)) and adv[0, 1] > 0,
        f"advantages={np.round(adv[0], 4).tolist()}"))

    rng = np.random.default_rng(1)
    n, k = 30, 4
    gold2 = (rng.random((n, k)) > 0.5).astype(float)
    same = gold2 * 0.9 + rng.random((n, k)) * 0.1
    pair = SnapshotPair(judge_start_on_y0=same, judge_end_on_y0=same,
                        judge_start_on_yk=same, judge_end_on_yk=same,
                        gold_on_y0=gold2, gold_on_yk=gold2)
    res = factorial_attribution(pair, n_resamples=64)
    checks.append(Check(
        "2x2 identity residual is exactly zero",
        res.interaction_identity_residual < 1e-12,
        f"residual={res.interaction_identity_residual:.2e}"))
    checks.append(Check(
        "frozen judge => judge-drift term identically zero",
        abs(res.terms["judge_drift(ΔJ=E10-E00)"].point) < 1e-12))
    return checks


# ---------------------------------------------------------------------------
# 3. statistical contracts
# ---------------------------------------------------------------------------


def _stat_checks() -> list[Check]:
    checks: list[Check] = []
    rng = np.random.default_rng(2)

    gold = (rng.random((100, 8)) > 0.5).astype(float)
    a = per_prompt_stats(gold * 0.7 + rng.random((100, 8)) * 0.3, gold)
    b = per_prompt_stats(gold * 0.1 + rng.random((100, 8)) * 0.9, gold)
    res = paired_cluster_bootstrap({"a": a, "b": b}, "acc_rank",
                                   [("a", "b")], n_resamples=400, seed=0)
    c = res.contrasts["a-b"]
    checks.append(Check(
        "paired contrast point == difference of cell points",
        abs(c.point - (res.cells["a"].point - res.cells["b"].point)) < 1e-12))
    checks.append(Check(
        "paired contrast has non-zero bootstrap spread",
        float(c.boot.std()) > 0,
        f"sd={float(c.boot.std()):.4f}"))

    eq = BootstrapCI(0.0, -0.01, 0.012, np.zeros(1))
    inc = BootstrapCI(0.04, -0.02, 0.11, np.zeros(1))
    go = BootstrapCI(0.05, 0.01, 0.09, np.zeros(1))
    checks.append(Check(
        "three-way triage: equivalence / inconclusive / go",
        classify_effect(eq, sesoi_pt=2.0) == "equivalence"
        and classify_effect(inc, sesoi_pt=2.0) == "inconclusive"
        and classify_effect(go, sesoi_pt=2.0) == "go"))

    bad = non_inferiority_test(np.full(300, -0.06), margin=0.025)
    ok = non_inferiority_test(np.full(300, 0.001), margin=0.025)
    checks.append(Check(
        "non-inferiority direction (6pt degradation fails, no change passes)",
        (not bad.non_inferior) and ok.non_inferior))

    ctrl = rng.normal(0.8, 0.02, 400)
    det = calibrate(ctrl, 0.05, "decrease")
    checks.append(Check(
        "matched-FPR threshold achieves the target rate on the control arm",
        abs(det.achieved_fpr - 0.05) < 0.02,
        f"achieved={det.achieved_fpr:.3f}"))
    checks.append(Check(
        "no false trigger on a flat (non-degrading) series",
        first_detection(np.full(10, 0.8), det, consecutive=2) is None))

    nan_ci = __import__("csr_grpo.diagnostics.bootstrap", fromlist=["x"]).percentile_ci(
        np.array([0.2, np.nan, 0.3]), point=0.25)
    checks.append(Check(
        "undefined bootstrap resamples are dropped and counted, not propagated",
        np.isfinite(nan_ci.lo) and nan_ci.n_dropped == 1))
    return checks


# ---------------------------------------------------------------------------
# 4. supervision invariant
# ---------------------------------------------------------------------------


def _supervision_checks() -> list[Check]:
    checks: list[Check] = []

    led = SupervisionLedger()
    blocked = False
    try:
        with led.training_scope():
            led.access(Level.L3, Role.AUDIT_REPORTING, "should fail")
    except SupervisionViolation:
        blocked = True
    checks.append(Check("L3 is unreachable inside a training scope", blocked))
    checks.append(Check("denied reads are not counted as gold calls",
                        led.gold_calls_in_training == 0))

    led2 = SupervisionLedger()
    with led2.training_scope(gold_control=True):
        led2.access(Level.L3, Role.E2_DIAGNOSTIC_GOLD_ARM, "declared arm")
    s = led2.summary()
    checks.append(Check(
        "declared gold-control reads are counted separately",
        s["gold_calls_in_training"] == 0 and s["gold_calls_in_gold_control"] == 1))

    led3 = SupervisionLedger()
    with led3.training_scope():
        led3.access(Level.L1, Role.METHOD_WARMUP, "demo")
        led3.access(Level.L2, Role.METHOD_TRAIN, "advantage")
    checks.append(Check("L1/L2 remain available to training",
                        led3.count(granted=True) == 2))

    # the CLI's own audit path must agree with the summary
    summary_keys = set(led3.summary())
    checks.append(Check(
        "ledger summary exposes both counters",
        bool({"gold_calls_in_training", "gold_calls_in_gold_control"} <= summary_keys)))
    return checks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_audit(*, run_pytest: bool = False) -> AuditReport:
    report = AuditReport(
        deliverables=audit_deliverables(),
        semantics=_metric_checks(),
        statistics=_stat_checks(),
        supervision=_supervision_checks(),
    )
    if run_pytest:
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q",
                               "--disable-warnings"], cwd=ROOT,
                              capture_output=True, text=True)
        tail = (proc.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        report.deliverables.append(Check(
            name="test suite passes", passed=proc.returncode == 0, detail=tail[0]))
    return report


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    report = run_audit(run_pytest="--with-pytest" in argv)
    print(json.dumps(report.as_dict(), indent=2))
    failed = [c for group in (report.deliverables, report.semantics,
                              report.statistics, report.supervision)
              for c in group if not c.passed]
    if failed:
        print(f"\nAUDIT FAILED: {len(failed)} check(s)", file=sys.stderr)
        return 1
    print("\nAUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
