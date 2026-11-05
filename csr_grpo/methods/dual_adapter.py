"""Dual-Adapter (+ Replay) — the paper's minimal structural repair (Row 8).

Two design choices matter and are both encoded here rather than in the runner:

1. **``r_S = 0``** removes the shared adapter entirely, so the judge view and the
   actor view cannot contaminate each other.  The 2x2 term
   ``E_10 − E_00`` (judge drift on fixed answers) is then *identically* zero,
   which is asserted in the tests.
2. **In-stage judge freezing.**  Under Dual-Adapter the judge is updated only
   during warm-up.  Contamination is therefore impossible by construction, and
   the only route for reward error to reach the Actor is via the current task's
   on-policy rollouts — exactly the measurement line E1/E2 care about.

Replay adds a cross-stage contrast warm-up: at the start of stage ``t`` the
judge is warmed on a 1:1 mixture of historical contrast pairs and fresh ones.
``ReplayBuffer`` owns the pairing and the sampling so the 1:1 ratio is exact
rather than "roughly".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..config.schema import Config
from ..methods.lora import AdapterSpec
from ..rewards.advantage import AdvMode
from ..rewards.sources import RewardSource
from .base import BaseMethod, MethodPlan


@dataclass
class DualAdapterMethod(BaseMethod):
    name: str = "dual_adapter"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    reward_source: RewardSource = RewardSource.SELF
    adv_mode: AdvMode = "grpo"

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name,
            spec=self.spec,
            judge_trainable_in_stage=False,
            replay=False,
            adv_mode=self.adv_mode,
            n_subpolicies=1,
            reward_source=self.reward_source,
            notes=(
                "r_S=0: judge and actor views share no parameters",
                "judge frozen inside a stage; E_10 - E_00 == 0 by construction",
            ),
        )

    @classmethod
    def from_config(cls, cfg: Config, *, adv_mode: AdvMode | None = None,
                    reward_source: RewardSource | None = None,
                    rename: str | None = None) -> "DualAdapterMethod":
        r = cfg.model.r_actor
        spec = AdapterSpec.dual_adapter(r=r, alpha=float(cfg.model.lora_alpha))
        return cls(
            name=rename or "dual_adapter",
            spec=spec,
            reward_source=reward_source or RewardSource(cfg.reward_source),
            adv_mode=adv_mode or cfg.train.adv_mode,
        )


@dataclass
class DualAdapterReplayMethod(DualAdapterMethod):
    name: str = "dual_adapter_replay"
    replay_ratio: float = 1.0
    replay_window: int | None = None

    def plan(self) -> MethodPlan:
        plan = super().plan()
        plan.name = self.name
        plan.replay = True
        plan.notes = plan.notes + (
            f"replay warm-up at 1:{self.replay_ratio:g} historical:fresh",
        )
        return plan

    @classmethod
    def from_config(cls, cfg: Config, *, adv_mode: AdvMode | None = None,
                    reward_source: RewardSource | None = None,
                    rename: str | None = None) -> "DualAdapterReplayMethod":
        r = cfg.model.r_actor
        spec = AdapterSpec.dual_adapter(r=r, alpha=float(cfg.model.lora_alpha))
        return cls(
            name=rename or "dual_adapter_replay",
            spec=spec,
            reward_source=reward_source or RewardSource(cfg.reward_source),
            adv_mode=adv_mode or cfg.train.adv_mode,
            replay_ratio=float(cfg.train.replay_ratio),
            replay_window=cfg.train.replay_window_stages,
        )


# ---------------------------------------------------------------------------
# replay buffer
# ---------------------------------------------------------------------------


@dataclass
class ContrastPair:
    """One (prompt, good answer, bad answer) triple.

    Route-A/route-B provenance is recorded so the paper can report per-operator
    true-negative rates and route-B yield without re-deriving provenance.
    """

    task_id: str
    prompt_ids: tuple[int, ...]
    good: int
    bad: int
    stage: int
    route: str = "A"                     # "A" rule perturbation | "B" consensus
    weak_label: float = 1.0              # 0.5 => ambiguous / soft
    verified_flip: bool = True           # route A abstention flag


class ReplayBuffer:
    """Cross-stage contrast storage with exact 1:1 sampling."""

    def __init__(self, ratio: float = 1.0, window: int | None = None):
        if ratio <= 0:
            raise ValueError("replay ratio must be positive")
        self.ratio = float(ratio)
        self.window = window
        self.pairs: list[ContrastPair] = []

    # -- ingest -------------------------------------------------------------
    def add_stage(self, pairs: Iterable[ContrastPair]) -> int:
        new = list(pairs)
        self.pairs.extend(new)
        return len(new)

    def prune(self, current_stage: int) -> None:
        if self.window is None:
            return
        lo = current_stage - self.window
        self.pairs = [p for p in self.pairs if p.stage >= lo]

    # -- sampling -----------------------------------------------------------
    def sample(self, n_fresh: int, rng: np.random.Generator,
               stage_filter: Iterable[int] | None = None) -> list[ContrastPair]:
        """Draw ``ratio * n_fresh`` historical pairs (no replacement per call)."""
        pool = self.pairs
        if stage_filter is not None:
            allowed = set(stage_filter)
            pool = [p for p in pool if p.stage in allowed]
        n_hist = int(round(self.ratio * n_fresh))
        if n_hist == 0 or not pool:
            return []
        idx = rng.choice(len(pool), size=min(n_hist, len(pool)), replace=False)
        return [pool[i] for i in idx]

    def stage_counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for p in self.pairs:
            out[p.stage] = out.get(p.stage, 0) + 1
        return dict(sorted(out.items()))

    def provenance(self) -> dict[str, float]:
        if not self.pairs:
            return {"n": 0}
        routes = {}
        for p in self.pairs:
            routes[p.route] = routes.get(p.route, 0) + 1
        flips = np.mean([p.verified_flip for p in self.pairs])
        ambiguous = np.mean([p.weak_label == 0.5 for p in self.pairs])
        return {
            "n": len(self.pairs),
            "route_A": routes.get("A", 0),
            "route_B": routes.get("B", 0),
            "flip_rate_after_perturbation": float(flips),
            "ambiguous_share": float(ambiguous),
        }

    def __len__(self) -> int:
        return len(self.pairs)


def build_warmup_batch(buffer: ReplayBuffer, fresh: Sequence[ContrastPair],
                       rng: np.random.Generator) -> list[ContrastPair]:
    """Exact 1:1 (or 1:ratio) mixture for the warm-up loss."""
    fresh = list(fresh)
    hist = buffer.sample(len(fresh), rng)
    return hist + fresh
