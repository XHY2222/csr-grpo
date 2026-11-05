"""Dual measurement line: one schedule drives both lines.

Line A — ``P_audit``: a frozen blind audit probe of the *old* stage, scored by
the stage-start and stage-end judge snapshots on frozen answer sets ``Y_old``.
Line B — ``P_on-policy``: the *current* stage's live rollouts, scored in the
same step.  Line B is what actually enters the policy gradient, so the paper
insists it be measured alongside line A; a degradation that appears only on A
never touched the Actor.

``DualLine`` owns the step-indexed accumulation so E1/E2 can pull aligned
series (same steps, same metric names) for their statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .fidelity import (
    DEFAULT_METRICS,
    FidelityReport,
    MetricName,
    PerPromptStats,
    evaluate_line,
    per_prompt_stats,
)

LINE_AUDIT = "P_audit"
LINE_ONPOLICY = "P_on-policy"


@dataclass
class LineSample:
    """One (line, step) measurement."""

    line: str
    step: int
    stats: PerPromptStats
    report: FidelityReport
    meta: dict[str, float | str] = field(default_factory=dict)


class DualLine:
    """Accumulates aligned line-A / line-B samples across a stage."""

    def __init__(self, metrics: tuple[MetricName, ...] = DEFAULT_METRICS,
                 tau: float = 0.05):
        self.metrics = metrics
        self.tau = tau
        self.samples: dict[str, list[LineSample]] = {LINE_AUDIT: [], LINE_ONPOLICY: []}

    # -- measurement --------------------------------------------------------
    def measure(self, line: str, step: int, judge: np.ndarray, gold: np.ndarray,
                **meta) -> LineSample:
        if line not in self.samples:
            raise ValueError(f"unknown line {line!r}")
        stats = per_prompt_stats(judge, gold, tau=self.tau)
        report = FidelityReport(
            values={m: _agg(stats, m) for m in self.metrics},
            n_prompts=len(stats),
            n_informative=int(stats.informative.sum()),
            n_degenerate=int(stats.degenerate.sum()),
            tau=self.tau,
        )
        sample = LineSample(line=line, step=step, stats=stats, report=report,
                            meta=dict(meta))
        self.samples[line].append(sample)
        return sample

    # -- retrieval ----------------------------------------------------------
    def series(self, line: str, metric: str) -> tuple[list[int], list[float]]:
        steps, vals = [], []
        for s in self.samples[line]:
            steps.append(s.step)
            vals.append(s.report.values.get(metric, float("nan")))
        return steps, vals

    def latest(self, line: str) -> LineSample | None:
        return self.samples[line][-1] if self.samples[line] else None

    def aligned(self, metric: str) -> dict[str, tuple[list[int], list[float]]]:
        return {line: self.series(line, metric) for line in self.samples}

    # -- discrepancy: does line A disagree with line B? ---------------------
    def divergence(self, metric: str = "sign_agree") -> float:
        """|A − B| at the latest step measured on *both* lines.

        Comparing each line's own latest sample is not the same thing: the two
        lines are sampled on the same schedule in principle, but a partial or
        interrupted run leaves them at different steps, and subtracting values
        from different moments reports a difference that belongs to the step
        gap rather than to the two probes.  ``nan`` when no step is shared.
        """
        common = (set(s.step for s in self.samples[LINE_AUDIT])
                  & set(s.step for s in self.samples[LINE_ONPOLICY]))
        if not common:
            return float("nan")
        step = max(common)
        a = next(s for s in reversed(self.samples[LINE_AUDIT]) if s.step == step)
        b = next(s for s in reversed(self.samples[LINE_ONPOLICY]) if s.step == step)
        return abs(a.report.values.get(metric, np.nan)
                   - b.report.values.get(metric, np.nan))

    # -- export -------------------------------------------------------------
    def table(self) -> list[dict]:
        rows = []
        for line, items in self.samples.items():
            for s in items:
                row = {"line": line, "step": s.step}
                row.update(s.report.as_row())
                rows.append(row)
        return rows


def _agg(stats: PerPromptStats, metric: str) -> float:
    from .fidelity import aggregate

    return aggregate(stats, metric)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# streaming snapshot bookkeeping for the 2x2 design
# ---------------------------------------------------------------------------


@dataclass
class SnapshotBook:
    """Holds the four score matrices needed for a stage's 2x2 table.

    The runner fills these as it goes: ``Y_0``/``Y_k`` are generated once per
    snapshot point, ``judge_start``/``judge_end`` are the two judge snapshots
    plus their gold labels (the latter obtained only through an L3 audit grant).
    """

    judge_start_on_y0: np.ndarray | None = None
    judge_end_on_y0: np.ndarray | None = None
    judge_start_on_yk: np.ndarray | None = None
    judge_end_on_yk: np.ndarray | None = None
    gold_on_y0: np.ndarray | None = None
    gold_on_yk: np.ndarray | None = None
    filled: dict[str, bool] = field(default_factory=dict)

    _FIELDS = ("judge_start_on_y0", "judge_end_on_y0", "judge_start_on_yk",
               "judge_end_on_yk", "gold_on_y0", "gold_on_yk")

    def set(self, name: str, value: np.ndarray) -> None:
        if name not in self._FIELDS:
            raise KeyError(name)
        setattr(self, name, value)
        self.filled[name] = True

    @property
    def complete(self) -> bool:
        return all(getattr(self, f) is not None for f in self._FIELDS)

    def missing(self) -> list[str]:
        return [f for f in self._FIELDS if getattr(self, f) is None]
