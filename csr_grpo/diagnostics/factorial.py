"""2x2 snapshot factorial attribution and the stage-boundary chain.

Definition (paper §2x2)
-----------------------
Two judges ``{θ_J^(t,0), θ_J^(t,k)}`` and two answer sets ``{Y_0, Y_k}``
sampled from the actor at the start / at step k of the stage, give four error
cells (error = ``1 − Acc_rank`` or ``1 − 1 − Sign_agree``)::

    E00 = E(θ_J^(t,0), Y_0)     E10 = E(θ_J^(t,k), Y_0)
    E01 = E(θ_J^(t,0), Y_k)     E11 = E(θ_J^(t,k), Y_k)

    E11 − E00 ≡ (E10 − E00) + (E01 − E00) + I_int
    I_int     = E11 − E10 − E01 + E00

``E10 − E00`` is judge drift on fixed answers (≡ 0 under a frozen judge),
``E01 − E00`` is the output-distribution-shift effect, ``I_int`` is the
non-additive co-evolution term.  The identity is exact per prompt, so the
decomposition is *not* an approximation and its residuals must be ~1e-12.

Boundary chain
--------------
The stage's ``θ_J^(t,0)`` has already been through replay warm-up, so an extra
boundary term ``E_w = E(θ_J^(t,0), Y_old) − E(θ_J^(t−1,end), Y_old)`` is
prepended.  It is also the H6 non-inferiority object.

Caveat kept from the plan: this module answers *where the error moved*, never
*why*; causality is E2's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .bootstrap import (
    BootstrapCI,
    paired_cluster_bootstrap,
    percentile_ci,
)
from .fidelity import PerPromptStats, aggregate, per_prompt_stats

ErrorFn = Callable[[PerPromptStats], np.ndarray]


# ---------------------------------------------------------------------------
# error extraction
# ---------------------------------------------------------------------------


def error_from(stats: PerPromptStats, kind: str = "acc_rank") -> np.ndarray:
    """Per-prompt error = 1 - metric; degenerate prompts stay NaN."""
    if kind == "acc_rank":
        return 1.0 - stats.acc_rank
    if kind == "sign_agree":
        return 1.0 - stats.sign_agree
    if kind == "rho_adv":
        return 1.0 - stats.rho_adv
    raise ValueError(kind)


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")


# ---------------------------------------------------------------------------
# the 2x2 table
# ---------------------------------------------------------------------------

_CELL_KEYS = ("E00", "E10", "E01", "E11")


@dataclass
class SnapshotPair:
    """Scores of two judges on two answer sets (same prompt order)."""

    judge_start_on_y0: np.ndarray
    judge_end_on_y0: np.ndarray
    judge_start_on_yk: np.ndarray
    judge_end_on_yk: np.ndarray
    gold_on_y0: np.ndarray
    gold_on_yk: np.ndarray
    prompt_ids: np.ndarray | None = None
    tau: float = 0.05
    eps: float = 1e-4

    def validate(self) -> None:
        shapes = {
            name: arr.shape
            for name, arr in {
                "judge_start_on_y0": self.judge_start_on_y0,
                "judge_end_on_y0": self.judge_end_on_y0,
                "judge_start_on_yk": self.judge_start_on_yk,
                "judge_end_on_yk": self.judge_end_on_yk,
                "gold_on_y0": self.gold_on_y0,
                "gold_on_yk": self.gold_on_yk,
            }.items()
        }
        n = self.judge_start_on_y0.shape[0]
        for name, shape in shapes.items():
            if shape[0] != n:
                raise ValueError(f"{name} has {shape[0]} prompts, expected {n}")
        if self.judge_start_on_y0.shape != self.gold_on_y0.shape:
            raise ValueError("judge/gold shape mismatch on Y_0")
        if self.judge_start_on_yk.shape != self.gold_on_yk.shape:
            raise ValueError("judge/gold shape mismatch on Y_k")

    def cells(self) -> dict[str, PerPromptStats]:
        self.validate()
        kw = dict(tau=self.tau, eps=self.eps)
        return {
            "E00": per_prompt_stats(self.judge_start_on_y0, self.gold_on_y0, **kw),
            "E10": per_prompt_stats(self.judge_end_on_y0, self.gold_on_y0, **kw),
            "E01": per_prompt_stats(self.judge_start_on_yk, self.gold_on_yk, **kw),
            "E11": per_prompt_stats(self.judge_end_on_yk, self.gold_on_yk, **kw),
        }


@dataclass
class TermCI:
    name: str
    point: float
    ci: BootstrapCI | None = None
    note: str = ""

    @property
    def significant(self) -> bool:
        return self.ci is not None and self.ci.excludes(0.0)

    def as_dict(self) -> dict:
        d = {"name": self.name, "point": self.point, "significant": self.significant,
             "note": self.note}
        if self.ci is not None:
            d["ci"] = [self.ci.lo, self.ci.hi]
        return d


@dataclass
class FactorialResult:
    kind: str
    terms: dict[str, TermCI]
    cell_means: dict[str, float]
    interaction_identity_residual: float
    n_prompts: int
    n_resamples: int = 0

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "cell_means": self.cell_means,
            "terms": {k: v.as_dict() for k, v in self.terms.items()},
            "identity_residual": self.interaction_identity_residual,
            "n_prompts": self.n_prompts,
            "n_resamples": self.n_resamples,
        }


def factorial_attribution(pair: SnapshotPair, *, kind: str = "acc_rank",
                          n_resamples: int = 2000, level: float = 0.95,
                          seed: int = 0) -> FactorialResult:
    """Compute the four cells, the three terms, and bootstrap CIs per term.

    Term CIs use a single paired resample so that the terms stay coherent
    (they are deterministic linear combinations of the same cells).
    """
    cells = pair.cells()
    err = {k: error_from(v, kind) for k, v in cells.items()}

    # exact identity per prompt
    resid = (err["E11"] - err["E00"]) - (
        (err["E10"] - err["E00"]) + (err["E01"] - err["E00"]) + 0.0
    ) - (err["E11"] - err["E10"] - err["E01"] + err["E00"])
    resid = float(np.nanmax(np.abs(resid))) if len(err["E00"]) else 0.0

    point = {k: _nanmean(v) for k, v in err.items()}
    delta_j = point["E10"] - point["E00"]
    delta_y = point["E01"] - point["E00"]
    delta_total = point["E11"] - point["E00"]
    interaction = delta_total - delta_j - delta_y

    # paired bootstrap over the same prompts
    rng = np.random.default_rng(seed)
    n = len(err["E00"])
    idx_boot = rng.integers(0, n, size=(n_resamples, n))

    def boot_term(fn: Callable[[np.ndarray], float]) -> np.ndarray:
        out = np.empty(n_resamples)
        for b, idx in enumerate(idx_boot):
            s = {k: v[idx] for k, v in err.items()}
            out[b] = fn(s)
        return out

    terms_raw = {
        "judge_drift(ΔJ=E10-E00)": (
            lambda s: _nanmean(s["E10"]) - _nanmean(s["E00"]), delta_j),
        "output_shift(ΔY=E01-E00)": (
            lambda s: _nanmean(s["E01"]) - _nanmean(s["E00"]), delta_y),
        "interaction(I_int)": (
            lambda s: (_nanmean(s["E11"]) - _nanmean(s["E00"]))
            - (_nanmean(s["E10"]) - _nanmean(s["E00"]))
            - (_nanmean(s["E01"]) - _nanmean(s["E00"])), interaction),
        "total(E11-E00)": (
            lambda s: _nanmean(s["E11"]) - _nanmean(s["E00"]), delta_total),
    }

    terms: dict[str, TermCI] = {}
    for name, (fn, p) in terms_raw.items():
        samples = boot_term(fn)
        terms[name] = TermCI(name=name, point=float(p),
                             ci=percentile_ci(samples, float(p), level))

    frozen_note = ""
    if abs(delta_j) < 1e-12:
        frozen_note = "judge frozen within stage (Dual-Adapter): ΔJ ≡ 0 by construction"
        terms["judge_drift(ΔJ=E10-E00)"].note = frozen_note

    return FactorialResult(
        kind=kind, terms=terms, cell_means=point,
        interaction_identity_residual=resid, n_prompts=n, n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# boundary chain
# ---------------------------------------------------------------------------


@dataclass
class BoundaryChain:
    """E_w prepended to the in-stage 2x2 terms.

    ``E_w`` compares the stage-start judge (post replay warm-up) against the
    previous stage's final judge on the *old* answer set, i.e. the same fixed
    answers for both judges.
    """

    E_w: TermCI
    in_stage: FactorialResult
    kind: str = "acc_rank"

    def flatten(self) -> dict[str, TermCI]:
        out = {"E_w(warmup boundary)": self.E_w}
        out.update(self.in_stage.terms)
        return out

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "chain": {k: v.as_dict() for k, v in self.flatten().items()},
            "in_stage": self.in_stage.as_dict(),
            "interpretation": _interpret(self),
        }

    def dominant_term(self, exclude_total: bool = True) -> str:
        items = [(k, abs(v.point)) for k, v in self.flatten().items()
                 if not (exclude_total and k.startswith("total"))]
        return max(items, key=lambda kv: kv[1])[0] if items else ""


def _interpret(chain: BoundaryChain) -> str:
    """Pre-registered branch interpretation (plan: '预注册诊断分支')."""
    terms = chain.flatten()
    total = terms.get("total(E11-E00)")
    if total is None or not np.isfinite(total.point) or total.point <= 0:
        return "no error growth in this stage on this line"
    dj = terms.get("judge_drift(ΔJ=E10-E00)")
    dy = terms.get("output_shift(ΔY=E01-E00)")
    if dj is not None and dj.ci is not None and dj.ci.lo > 0 and \
            abs(dj.point) >= abs(dy.point if dy else 0.0):
        return "judge drift dominates -> parameter contamination hypothesis"
    if dy is not None and dy.ci is not None and (
            abs(dj.point) < 1e-3 or dj.ci.lo <= 0) and abs(dy.point) > 0:
        return ("output-distribution shift dominates -> cross-role gradient "
                "contamination is limited; focus on output drift / ΔÂ_canon")
    return "no single term dominates; report the full decomposition"


def boundary_attribution(
    judge_prev_end_on_old: np.ndarray,
    judge_stage_start_on_old: np.ndarray,
    gold_on_old: np.ndarray,
    pair: SnapshotPair,
    *,
    kind: str = "acc_rank",
    n_resamples: int = 2000,
    seed: int = 0,
) -> BoundaryChain:
    """Compute ``E_w`` and attach it to the in-stage factorial."""
    e_prev = error_from(per_prompt_stats(judge_prev_end_on_old, gold_on_old,
                                         tau=pair.tau), kind)
    e_now = error_from(per_prompt_stats(judge_stage_start_on_old, gold_on_old,
                                        tau=pair.tau), kind)
    diff = e_now - e_prev
    point = _nanmean(diff)
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = np.array([_nanmean(diff[i]) for i in idx])
    e_w = TermCI("E_w(warmup boundary)", point,
                 percentile_ci(samples, point, 0.95),
                 "H6 non-inferiority object: expect Δ ≤ +δ_NI")
    in_stage = factorial_attribution(pair, kind=kind, n_resamples=n_resamples,
                                     seed=seed + 1)
    return BoundaryChain(E_w=e_w, in_stage=in_stage, kind=kind)


# ---------------------------------------------------------------------------
# convenience: batch of prompts across a trajectory
# ---------------------------------------------------------------------------


@dataclass
class FactorialTrajectory:
    """Factorial results at several steps; used for Figure 2/3(a)."""

    steps: list[int] = field(default_factory=list)
    results: list[FactorialResult] = field(default_factory=list)

    def append(self, step: int, res: FactorialResult) -> None:
        self.steps.append(step)
        self.results.append(res)

    def series(self) -> dict[str, list[float]]:
        if not self.results:
            return {}
        keys = list(self.results[0].terms)
        return {k: [r.terms[k].point for r in self.results] for k in keys}

    def meta(self) -> dict[str, list[int]]:
        return {"steps": self.steps}
