"""Method layer: what each baseline actually changes.

A method is described declaratively (``MethodPlan``) so the runner never needs
per-method branches, and so the supervision story is inspectable: a plan that
asks for a gold reward is rejected at construction unless it is a diagnostic
arm or the Row-10 reference.

Correspondence to the paper's run matrix
----------------------------------------
``shared_grpo``        Row 3      one shared adapter, judge updated in-stage
``dual_adapter``       Row 8/C    ``r_S = 0``, judge frozen in-stage
``dual_adapter_replay``Row 8      + cross-stage contrast replay warm-up
``rler_faithful``      Row 5a     M=2 sub-policies, matched rollout budget
``judge_lora_ensemble``Row 5a'    single actor, K judge adapters voting
``fisher_mask``        Row 7a     fixed architecture, Fisher-coordinate mask
``random_mask``        Row 7b     same mask count, stratified random positions
``gopo_rank_only``     ablation f rank-only advantage
``gold_reference``     Row 10     reference only, never a "method"
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..config.schema import Config
from ..methods.lora import AdapterSpec
from ..rewards.advantage import AdvantageConfig, AdvMode
from ..rewards.sources import RewardSource


@dataclass
class MethodPlan:
    """Everything the runner needs to know about a method."""

    name: str
    spec: AdapterSpec
    judge_trainable_in_stage: bool
    replay: bool = False
    adv_mode: AdvMode = "grpo"
    degenerate_zero: bool = False
    n_subpolicies: int = 1              # RLER M
    judge_ensemble_k: int = 1
    mask_fraction: float = 0.0
    mask_kind: str = "none"             # none | fisher | random
    reward_source: RewardSource = RewardSource.SELF
    # supervised baselines (Rows 1/2a/2b) never take a policy-gradient step
    supervised_only: bool = False
    # Row 4: judge is frozen and conditioned in-context; warm-up is skipped
    icl_judge: bool = False
    icl_shots: int = 6
    # Row 9: train on a mixture of all stages in every step
    static_mix: bool = False
    notes: tuple[str, ...] = ()

    @property
    def uses_gold_reward(self) -> bool:
        return self.reward_source is RewardSource.GOLD

    def advantage_config(self, eps: float, clip: float | None = 5.0) -> AdvantageConfig:
        return AdvantageConfig(
            mode=self.adv_mode,
            eps=eps,
            degenerate_zero=self.degenerate_zero,
            degenerate_source="judge",
            clip=clip,
        )


class Method(Protocol):
    name: str

    def plan(self) -> MethodPlan: ...


# ---------------------------------------------------------------------------
# shared implementation
# ---------------------------------------------------------------------------


@dataclass
class BaseMethod:
    name: str = "base"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)

    def plan(self) -> MethodPlan:  # pragma: no cover - overridden
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: Config) -> "BaseMethod":
        raise NotImplementedError

    def capacity_report(self, model) -> dict[str, float]:
        """Capacity proxies required when comparing masked architectures.

        ``shared`` lives on the model (one object seen by both views), while
        ``actor``/``judge`` live on the adapter banks, so the two cases are
        resolved separately.
        """
        out: dict[str, float] = {}
        for name in self.spec.actor_active:
            if name == "shared":
                mod = getattr(model, "shared", None)
            else:
                mod = getattr(getattr(model, "actor_bank", None), name, None)
            if mod is None or getattr(mod, "A", None) is None:
                out[f"actor.{name}.eff_rank"] = 0.0
                out[f"actor.{name}.fro"] = 0.0
                continue
            w = mod.effective_weight().detach()
            s = torch.linalg.svdvals(w)
            p = (s ** 2) / (s ** 2).sum()
            out[f"actor.{name}.eff_rank"] = float(
                torch.exp(-(p * torch.log(p + 1e-12)).sum()))
            out[f"actor.{name}.fro"] = float(w.norm())
        return out


def make_method(cfg: Config) -> BaseMethod:
    """Factory keyed on ``cfg.method``; the single place methods are wired."""
    from .dual_adapter import DualAdapterMethod, DualAdapterReplayMethod
    from .masking import FisherMaskMethod, StratifiedRandomMaskMethod
    from .rler import JudgeEnsembleMethod, RLERFaithfulMethod, SharedGRPOMethod

    name = cfg.method
    if name == "shared_grpo":
        return SharedGRPOMethod.from_config(cfg)
    if name == "dual_adapter":
        return DualAdapterMethod.from_config(cfg)
    if name == "dual_adapter_replay":
        return DualAdapterReplayMethod.from_config(cfg)
    if name == "rler_faithful":
        return RLERFaithfulMethod.from_config(cfg)
    if name == "judge_lora_ensemble":
        return JudgeEnsembleMethod.from_config(cfg)
    if name == "fisher_mask":
        return FisherMaskMethod.from_config(cfg)
    if name == "random_mask":
        return StratifiedRandomMaskMethod.from_config(cfg)
    if name == "gopo_rank_only":
        return DualAdapterReplayMethod.from_config(cfg, adv_mode="rank_only",
                                                   rename="gopo_rank_only")
    if name == "gold_reference":
        return DualAdapterReplayMethod.from_config(cfg, reward_source=RewardSource.GOLD,
                                                   rename="gold_reference")
    if name == "meta_rewarding":
        from .co_rewarding import MetaRewardingMethod

        return MetaRewardingMethod.from_config(cfg)
    if name == "online_judge_cotraining":
        from .co_rewarding import OnlineJudgeCoTrainingMethod

        return OnlineJudgeCoTrainingMethod.from_config(cfg)
    if name == "mgrpo_adapted":
        from .co_rewarding import MGRPOAdaptedMethod

        return MGRPOAdaptedMethod.from_config(cfg)
    if name == "seq_sft":
        from .baselines import SequentialSFTMethod

        return SequentialSFTMethod.from_config(cfg)
    if name == "sdft_only":
        from .baselines import SDFTMethod

        return SDFTMethod.from_config(cfg, icl=False)
    if name == "sdft_icl":
        from .baselines import SDFTMethod

        return SDFTMethod.from_config(cfg, icl=True)
    if name == "icl_judge":
        from .baselines import ICLJudgeMethod

        return ICLJudgeMethod.from_config(cfg)
    if name == "static_mix":
        from .baselines import StaticMixingMethod

        return StaticMixingMethod.from_config(cfg)
    raise ValueError(f"unknown method {name!r}")


def plan_table(methods: Sequence[BaseMethod]) -> list[dict]:
    rows = []
    for m in methods:
        p = m.plan()
        rows.append({
            "method": p.name,
            "r_shared": p.spec.r_shared,
            "r_actor": p.spec.r_actor_private,
            "r_judge": p.spec.r_judge_private,
            "judge_trainable_in_stage": p.judge_trainable_in_stage,
            "replay": p.replay,
            "adv_mode": p.adv_mode,
            "degenerate_zero": p.degenerate_zero,
            "M_subpolicies": p.n_subpolicies,
            "judge_ensemble_k": p.judge_ensemble_k,
            "mask": f"{p.mask_kind}:{p.mask_fraction:.0%}",
            "reward_source": str(p.reward_source),
        })
    return rows
