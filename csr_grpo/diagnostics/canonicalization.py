"""Controlled-canonicalisation probe: ΔÂ_canon and FlipRate (H7).

What this measures
------------------
Take a group of answers ``{y_1 … y_G}`` and a *semantically invariant* rewrite
of each ``{ỹ_1 … ỹ_G}``.  Recompute the judge scores on both and ask how far the
group-standardised advantages moved::

    ΔÂ_canon = E_x [ (1/G) Σ_i |Â_i(y) − Â_i(ỹ)| ]
    FlipRate = fraction of within-group orderings that reversed

The mathematical property the paper relies on: if a rewrite shifts every score
in a group by the same constant, standardisation cancels it and ``ΔÂ_canon = 0``.
That is why the metric is reported instead of the raw score difference.

What it does *not* prove
------------------------
A rise in ``ΔÂ_canon`` means "the ordering is sensitive to surface form".  It
does not prove the actor is gaming the judge, and it is not causal — H7 is a
*candidate* warning signal whose value is decided by held-out FPR/FNR
(plan §Component 3 · L1).  The docs and the result object both say so.

Denominator-collapse guard (plan L1)
-----------------------------------
When group reward variance → 0, tiny raw differences standardise into large
advantage moves; the metric would then be measuring the shrinking denominator.
So we also return the **unstandardised** shift ``Δr_canon`` and the ``σ_r``
trajectory, and drop groups below ``sigma_min``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from .fidelity import advantages

Domain = Literal["code", "ifeval", "math"]


# ---------------------------------------------------------------------------
# whitelist transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """One whitelisted surface rewrite.

    ``breaks_if`` lists the constraint types the rewrite would *violate*, which
    is the direction that matters: a transform is admissible only when the
    prompt's constraint set is disjoint from it.  (Getting this backwards would
    admit a word-adding rewrite on a prompt with a word-count constraint — the
    exact failure the plan's compatibility matrix exists to prevent.)
    """

    name: str
    domain: Domain
    apply: Callable[[str], str]
    breaks_if: frozenset[str] = frozenset()

    def __call__(self, text: str) -> str:
        return self.apply(text)


def _code_drop_comments(code: str) -> str:
    out = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _code_blank_lines(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.splitlines() if line.strip())


def _code_duplicate_semicolon(code: str) -> str:
    return code.replace(";\n", ";\n\n").replace("; ", ";  ")


def _math_transition(text: str) -> str:
    """Swap only non-logical step transitions; digits and operators untouched."""
    table = {
        # NOTE: "Step 1:" is deliberately absent — it carries a digit, and the
        # whitelist locks every digit. Only digit-free transitions are mapped.
        "Therefore,": "Thus,",
        "So,": "Hence,",
        "We get": "We obtain",
        "This gives": "This yields",
    }
    for a, b in table.items():
        text = text.replace(a, b)
    return text


def _ifeval_politeness(text: str) -> str:
    return "Certainly, " + text if not text.startswith("Certainly, ") else text


def _ifeval_blank_lines(text: str) -> str:
    return text.replace("\n\n", "\n\n ") if "\n\n" in text else text


CODE_TRANSFORMS: tuple[Transform, ...] = (
    Transform("drop_comment", "code", _code_drop_comments),
    Transform("collapse_blank_lines", "code", _code_blank_lines),
    Transform("pad_semicolon", "code", _code_duplicate_semicolon),
)

MATH_TRANSFORMS: tuple[Transform, ...] = (
    Transform("transition_word", "math", _math_transition),
)

# IFEval: the transform is admissible only if the prompt's constraint set does
# not include one it would break.  The matrix below is the machine-readable
# version of the plan's "算子 × 约束类型兼容矩阵".
# Operator x constraint compatibility matrix (the plan's 算子 × 约束类型兼容矩阵).
# Keys are transform names; values are the constraints that forbid them.
IFEVAL_COMPAT: dict[str, frozenset[str]] = {
    # adds words -> invalid under any counting constraint, and the added word
    # itself could violate a forbidden-word or casing rule
    "polite_prefix": frozenset({"word_count", "char_count", "forbidden_words",
                                "case", "keywords"}),
    # whitespace-only -> survives counting (word count unchanged) and casing,
    # but a literal newline could violate a restriction on blank lines
    "blank_lines": frozenset({"no_blank_lines", "format"}),
}

IFEVAL_TRANSFORMS: tuple[Transform, ...] = (
    Transform("polite_prefix", "ifeval", _ifeval_politeness,
              breaks_if=IFEVAL_COMPAT["polite_prefix"]),
    Transform("blank_lines", "ifeval", _ifeval_blank_lines,
              breaks_if=IFEVAL_COMPAT["blank_lines"]),
)


def transforms_for(domain: Domain) -> tuple[Transform, ...]:
    if domain == "code":
        return CODE_TRANSFORMS
    if domain == "math":
        return MATH_TRANSFORMS
    if domain == "ifeval":
        return IFEVAL_TRANSFORMS
    raise ValueError(domain)


def admissible(transform: Transform, constraints: Iterable[str]) -> bool:
    """True when none of the prompt's constraints would be broken by the rewrite.

    Code and math transforms carry an empty ``breaks_if``: they never touch
    digits, operators, logical connectives, or program semantics, so no
    constraint of those domains can be violated.  Their safety is instead
    verified empirically (sandbox output equality / label-preservation audit),
    which is what the plan asks for.
    """
    cset = set(constraints)
    return cset.isdisjoint(transform.breaks_if)


def select_transform(domain: Domain, constraints: Iterable[str] = ()
                    ) -> Transform | None:
    """First admissible transform, or ``None`` (the prompt is then excluded)."""
    for t in transforms_for(domain):
        if admissible(t, constraints):
            return t
    return None


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@dataclass
class CanonReport:
    """ΔÂ_canon and its guard rails for one probe batch."""

    delta_adv: float
    delta_raw: float
    flip_rate: float
    sigma_r_mean: float
    n_groups: int
    n_excluded_small_sigma: int
    coverage: float
    domain: Domain
    interpretation: str = field(default="")

    def as_dict(self) -> dict:
        return {
            "delta_adv_canon": self.delta_adv,
            "delta_raw_canon": self.delta_raw,
            "flip_rate": self.flip_rate,
            "sigma_r_mean": self.sigma_r_mean,
            "n_groups": self.n_groups,
            "n_excluded_small_sigma": self.n_excluded_small_sigma,
            "coverage": self.coverage,
            "domain": self.domain,
            "interpretation": self.interpretation,
        }


def _rank_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort")


def _flip_fraction(before: np.ndarray, after: np.ndarray) -> float:
    """Fraction of pairwise orderings that reversed."""
    n = len(before)
    if n < 2:
        return float("nan")
    rb, ra = _rank_order(before), _rank_order(after)
    flips = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if np.sign(rb[i] - rb[j]) != np.sign(ra[i] - ra[j]):
                flips += 1
    return flips / total if total else float("nan")


def compute_canon(original: np.ndarray, rewritten: np.ndarray, *,
                  sigma_min: float = 1e-3, eps: float = 1e-4,
                  domain: Domain = "code") -> CanonReport:
    """``ΔÂ_canon``, ``Δr_canon``, FlipRate over a ``(P, G)`` score batch.

    ``original`` and ``rewritten`` are judge scores on the two answer sets.
    Groups with ``σ_r < sigma_min`` are dropped *and counted*, so a shrinking
    denominator cannot masquerade as sensitivity.
    """
    o = np.asarray(original, dtype=np.float64)
    r = np.asarray(rewritten, dtype=np.float64)
    if o.shape != r.shape:
        raise ValueError(f"shape mismatch: {o.shape} vs {r.shape}")
    if o.ndim != 2:
        raise ValueError("expected (P, G)")

    adv_deltas: list[float] = []
    raw_deltas: list[float] = []
    flips: list[float] = []
    sigmas: list[float] = []
    excluded = 0

    for g in range(len(o)):
        sigma = float(np.std(o[g]))
        if sigma < sigma_min:
            excluded += 1
            continue
        sigmas.append(sigma)
        ao, ar = advantages(o[g:g + 1], eps=eps)[0], advantages(r[g:g + 1], eps=eps)[0]
        adv_deltas.append(float(np.mean(np.abs(ao - ar))))
        raw_deltas.append(float(np.mean(np.abs(o[g] - r[g]))))
        flips.append(_flip_fraction(o[g], r[g]))

    kept = len(adv_deltas)
    coverage = kept / len(o) if len(o) else float("nan")
    report = CanonReport(
        delta_adv=float(np.mean(adv_deltas)) if adv_deltas else float("nan"),
        delta_raw=float(np.mean(raw_deltas)) if raw_deltas else float("nan"),
        flip_rate=float(np.nanmean(flips)) if flips else float("nan"),
        sigma_r_mean=float(np.mean(sigmas)) if sigmas else float("nan"),
        n_groups=kept,
        n_excluded_small_sigma=excluded,
        coverage=float(coverage),
        domain=domain,
    )
    report.interpretation = _interpret(report)
    return report


def _interpret(rep: CanonReport) -> str:
    if not np.isfinite(rep.delta_adv):
        return "no admissible groups; metric undefined (report coverage, not zero)"
    if rep.coverage < 0.5:
        return (f"coverage {rep.coverage:.0%} — most groups excluded (σ_r < σ_min); "
                "do not read a sensitivity claim from this batch")
    if rep.delta_adv < 1e-6:
        return ("ΔÂ_canon ≈ 0: the rewrite moved every score by a common constant, "
                "so standardised advantages are invariant")
    if rep.delta_raw < rep.delta_adv * 0.5:
        return ("ΔÂ_canon large while Δr_canon small: possible denominator "
                "collapse — inspect σ_r before claiming sensitivity")
    return ("orderings react to surface form; this is a *candidate* warning "
            "signal only — H7 is decided by held-out FPR/FNR")


# ---------------------------------------------------------------------------
# thresholding / early warning (Stage 1–2 calibration, Stage 3–4 blind test)
# ---------------------------------------------------------------------------


@dataclass
class CanonDetector:
    mu: float
    sigma: float
    k: float = 2.0
    consecutive: int = 2

    @property
    def threshold(self) -> float:
        return self.mu + self.k * self.sigma

    @classmethod
    def calibrate(cls, anchor_values: Sequence[float], k: float = 2.0,
                  consecutive: int = 2) -> "CanonDetector":
        v = np.asarray(list(anchor_values), dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError("no anchor values to calibrate on")
        return cls(mu=float(v.mean()), sigma=float(v.std()), k=k,
                   consecutive=consecutive)

    def first_trigger(self, series: Sequence[float],
                      steps: Sequence[int] | None = None) -> int | None:
        run = 0
        for i, v in enumerate(series):
            if np.isfinite(v) and v > self.threshold:
                run += 1
                if run >= self.consecutive:
                    return int(steps[i]) if steps is not None else i
            else:
                run = 0
        return None


def evaluate_warning(trigger_step: int | None, peak_step: int | None,
                     horizon: int) -> dict:
    """Lead/lag relative to the gold peak, plus FPR/FNR bookkeeping.

    A "miss" (no trigger before the horizon) counts as a false negative; a
    trigger after the peak counts as a false positive for *warning* purposes but
    is still reported with its lag, since the plan asks for the distribution.
    """
    if peak_step is None:
        return {"status": "no_gold_peak"}
    if trigger_step is None:
        return {"status": "miss", "false_negative": True,
                "lead": None, "peak_step": peak_step}
    lead = peak_step - trigger_step
    if lead >= 0:
        return {"status": "early", "false_negative": False,
                "false_positive": False, "lead": int(lead),
                "peak_step": peak_step, "trigger_step": int(trigger_step)}
    return {"status": "late", "false_negative": True, "false_positive": True,
            "lead": int(lead), "peak_step": peak_step,
            "trigger_step": int(trigger_step)}


# ---------------------------------------------------------------------------
# empirical invariance checks (the plan's "construction + audit" requirement)
# ---------------------------------------------------------------------------


def verify_code_transform(original: str, rewritten: str,
                          samples: Sequence[str] = (),
                          run: Callable[[str, str], str] | None = None
                          ) -> dict:
    """Whitelisted code rewrites must be output-identical on test inputs.

    The plan is explicit that a rewriting is admitted only if it changes nothing
    observable: "改写前后必须在自生成测试输入上 100% 输出一致才纳入计算".  This
    function performs that check so a caller can drop non-invariant pairs
    instead of trusting the transform.

    ``run(code, input_text) -> stdout``; when omitted the check is reported as
    ``unverified`` rather than silently passing.
    """
    if run is None:
        return {"verified": False, "reason": "no sandbox runner supplied",
                "n_samples": 0, "n_mismatch": 0}
    mismatches = 0
    for s in samples:
        try:
            if run(original, s) != run(rewritten, s):
                mismatches += 1
        except Exception:
            mismatches += 1
    return {"verified": mismatches == 0,
            "reason": "output mismatch" if mismatches else "identical on all samples",
            "n_samples": len(samples), "n_mismatch": mismatches}


def label_preservation(originals: Sequence[str], rewrites: Sequence[str],
                       label_fn: Callable[[str], object], *,
                       audit_size: int = 100) -> dict:
    """Agreement rate of a label function before/after rewriting.

    Used for the math/logic transition-word transform, where the plan asks for a
    hand audit of 100 pairs per stage.  Here the audit is automated but reported
    with the same statistic, so a human sample can be compared against it.
    """
    pairs = list(zip(originals, rewrites))[:audit_size]
    if not pairs:
        return {"n_audited": 0, "preservation_rate": float("nan"),
                "changed": []}
    changed = [i for i, (a, b) in enumerate(pairs) if label_fn(a) != label_fn(b)]
    return {
        "n_audited": len(pairs),
        "preservation_rate": 1.0 - len(changed) / len(pairs),
        "changed": changed,
    }
