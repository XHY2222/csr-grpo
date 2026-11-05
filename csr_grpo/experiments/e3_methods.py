"""E3 — minimal method benefit (H3/C3).

Two-stage (Stage 1 → 2), same start, four arms:

1. ``dual_adapter``            — architecture only, no replay (Group C reuse)
2. ``dual_adapter_replay``     — the paper's Ours-Core (Row 8)
3. ``rler_faithful``           — M=2 sub-policies with matched rollout budget
4. ``judge_lora_ensemble``     — K judge adapters (our own ablation, not RLER)

Pre-registered gate: ``dual_adapter_replay`` must improve the forgetting measure
by ≥ 2 points over the shared baseline **and** be non-inferior (within 1.5 pt)
to ``rler_faithful`` on forward accuracy.  If E1/E2 hold but E3 fails, the paper
drops the architecture claim instead of the mechanism claim — that rule is
encoded in ``gate_result`` so a run cannot quietly reinterpret it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..compat import to_np, to_tensor
from ..config.schema import Config
from ..methods.base import make_method
from ..rewards.advantage import compute_advantages
from ..rewards.sources import RewardSource
from ..supervision.ledger import Level, Role, SupervisionLedger
from ..training.runner import Runner
from ..training.state import RolloutBatch


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def greedy_accuracy(runner: Runner, prompt_ids: torch.Tensor, stage: int) -> float:
    """Accuracy of the actor's modal answer (deterministic, high variance avoided)."""
    runner.model.set_stage(stage)
    with torch.no_grad():
        logits = runner.model.actor_logits(prompt_ids)
        pred = logits.argmax(dim=-1)
    gold = to_np(runner.score_with_gold(stage, prompt_ids, pred))
    return float(gold.mean())


def forgetting(acc_after: dict[str, float], acc_at_learning: dict[str, float]
               ) -> float:
    """Average drop on every stage seen so far (higher = more forgetting)."""
    if not acc_after:
        return float("nan")
    drops = [acc_at_learning[s] - acc_after[s] for s in acc_after]
    return float(np.mean(drops))


# ---------------------------------------------------------------------------
# training one arm
# ---------------------------------------------------------------------------


@dataclass
class ArmOutcome:
    arm: str
    acc_per_stage: dict[str, float] = field(default_factory=dict)
    acc_at_learning: dict[str, float] = field(default_factory=dict)
    forward_acc: float = float("nan")
    forgetting: float = float("nan")
    gold_reads: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "acc_per_stage": self.acc_per_stage,
            "acc_at_learning": self.acc_at_learning,
            "forward_acc": self.forward_acc,
            "forgetting": self.forgetting,
            "gold_reads": self.gold_reads,
            **self.extra,
        }


def _judge_train_step(runner: Runner, ctx, step: int) -> None:
    """Update the judge on the current stage (route-A style synthetic labels)."""
    if runner.judge_optimizer is None:
        return
    runner.warmup_judge(ctx.stage, n_rounds=1, n_per_round=16)


def run_arm_stages(cfg: Config, method_name: str, ledger: SupervisionLedger,
                   *, stages: Sequence[int] = (1, 2), steps_per_stage: int = 60,
                   n_eval: int = 32, seed: int = 0,
                   eval_n_prompts: int = 400) -> ArmOutcome:
    """Run a method through ``stages`` and measure forward accuracy and forgetting."""
    cfg = copy.deepcopy(cfg)
    cfg.method = method_name
    cfg.seed = seed
    cfg.stages_to_run = tuple(stages)
    if cfg.method == "gold_reference":
        cfg.is_reference = True
    cfg.validate()

    method = make_method(cfg)
    runner = Runner(cfg, method.plan(), ledger)
    acc_at_learning: dict[str, float] = {}
    eval_sets: dict[int, torch.Tensor] = {}

    for stage in stages:
        name = _stage_name(stage)
        eval_sets[stage] = runner.stage_prompts(stage, eval_n_prompts,
                                                seed=777 + stage)[0]
        runner.warmup_judge(stage, n_rounds=cfg.train.warmup_rounds)
        ctx = runner.make_context(stage, n_prompts=cfg.data.n_prompts_per_stage,
                                  seed=100 * stage,
                                  audit_stage=stage - 1 if stage > 1 else None,
                                  audit_seed=cfg.exp.audit_seed)
        acc_at_learning[name] = greedy_accuracy(runner, eval_sets[stage], stage)

        supervised = method.plan().supervised_only
        for step in range(1, steps_per_stage + 1):
            if supervised:
                # Row 2b distils an ICL-conditioned context; Row 2a/1 do not.
                is_icl = getattr(method, "icl", False)
                is_sdft = getattr(method, "name", "") in {"sdft_only", "sdft_icl"}
                runner.supervised_step(ctx, step, icl=is_icl,
                                       self_distill=is_sdft,
                                       n_shots=getattr(method, "n_shots", 6))
            else:
                runner.train_step(ctx, step)
                if method.plan().judge_trainable_in_stage:
                    _judge_train_step(runner, ctx, step)

    acc_per_stage = {}
    for stage in stages:
        name = _stage_name(stage)
        acc_per_stage[name] = greedy_accuracy(runner, eval_sets[stage], stage)

    forward = float(np.mean([acc_per_stage[_stage_name(s)] for s in stages]))
    forget = forgetting(acc_per_stage, acc_at_learning)
    return ArmOutcome(arm=method_name, acc_per_stage=acc_per_stage,
                      acc_at_learning=acc_at_learning,
                      forward_acc=forward, forgetting=forget,
                      extra={"capacity": method.capacity_report(runner.model)})


def _stage_name(stage: int) -> str:
    from ..backends.toy import STAGE_NAMES

    return STAGE_NAMES[stage - 1]


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


@dataclass
class E3Gate:
    fm_gain_vs_shared: float          # pt, positive = less forgetting
    forward_gap_vs_rler: float        # pt, positive = replay is better
    min_fm_gain_pt: float = 2.0
    max_forward_gap_pt: float = 1.5
    passed: bool = False
    interpretation: str = ""

    @classmethod
    def evaluate(cls, arms: dict[str, ArmOutcome], *,
                 proposed: str = "dual_adapter_replay",
                 baseline: str = "shared_grpo",
                 reference: str = "rler_faithful",
                 min_fm_gain_pt: float = 2.0,
                 max_gap_pt: float = 1.5) -> "E3Gate":
        base = arms.get(baseline) or arms.get("dual_adapter")
        ref = arms.get(reference)
        prop = arms.get(proposed)
        if prop is None:
            # the proposed arm was not run; the gate cannot evaluate
            return cls(passed=False, fm_gain_vs_shared=float("nan"),
                       forward_gap_vs_rler=float("nan"),
                       interpretation=f"proposed arm {proposed!r} not in arms")
        fm_gain = (base.forgetting - prop.forgetting) * 100 if base else float("nan")
        fwd_gap = ((prop.forward_acc - ref.forward_acc) * 100 if ref else float("nan"))
        passed = bool(np.isfinite(fm_gain) and fm_gain >= min_fm_gain_pt
                      and (not np.isfinite(fwd_gap) or fwd_gap >= -max_gap_pt))
        if passed:
            interp = ("H3 supported: minimal repair reduces forgetting and is "
                      "non-inferior on forward accuracy")
        else:
            interp = ("H3 not supported: drop the architecture-superiority claim "
                      "but keep C1/C2 mechanism findings if E1/E2 passed")
        return cls(fm_gain_vs_shared=fm_gain, forward_gap_vs_rler=fwd_gap,
                   min_fm_gain_pt=min_fm_gain_pt,
                   max_forward_gap_pt=max_gap_pt, passed=passed,
                   interpretation=interp)

    def as_dict(self) -> dict:
        return {
            "fm_gain_vs_shared_pt": self.fm_gain_vs_shared,
            "forward_gap_vs_rler_pt": self.forward_gap_vs_rler,
            "thresholds": {"fm_gain_pt": self.min_fm_gain_pt,
                           "forward_gap_pt": self.max_forward_gap_pt},
            "passed": self.passed,
            "interpretation": self.interpretation,
        }


@dataclass
class E3Result:
    arms: dict[str, dict]
    gate: dict

    def as_dict(self) -> dict:
        return {"arms": self.arms, "gate": self.gate}


def _avg_outcomes(outcomes: list[ArmOutcome]) -> ArmOutcome:
    """Average scalar metrics across seed-replicate ArmOutcomes."""
    if len(outcomes) == 1:
        return outcomes[0]
    import numpy as _np
    arm = outcomes[0].arm
    # per-stage accuracy: average across seeds
    all_stages = list(outcomes[0].acc_per_stage.keys())
    acc_per_stage = {
        s: float(_np.mean([o.acc_per_stage.get(s, float("nan")) for o in outcomes]))
        for s in all_stages
    }
    acc_at_learning = {
        s: float(_np.mean([o.acc_at_learning.get(s, float("nan")) for o in outcomes]))
        for s in all_stages
    }
    forward = float(_np.nanmean([o.forward_acc for o in outcomes]))
    forget = float(_np.nanmean([o.forgetting for o in outcomes]))
    return ArmOutcome(arm=arm, acc_per_stage=acc_per_stage,
                      acc_at_learning=acc_at_learning,
                      forward_acc=forward, forgetting=forget,
                      extra=outcomes[0].extra)


def run_e3(cfg: Config, ledger: SupervisionLedger, *, stages: Sequence[int] = (1, 2),
           steps_per_stage: int = 60, seed: int = 0,
           n_seeds: int = 1,
           arms: Sequence[str] | None = None) -> E3Result:
    arms = arms or tuple(cfg.exp.e3_arms) or ("shared_grpo", "dual_adapter",
                                              "dual_adapter_replay",
                                              "rler_faithful", "judge_lora_ensemble")
    outcomes: dict[str, ArmOutcome] = {}
    for name in arms:
        seed_runs = [
            run_arm_stages(cfg, name, ledger, stages=stages,
                           steps_per_stage=steps_per_stage,
                           seed=seed + s * 100)
            for s in range(n_seeds)
        ]
        outcomes[name] = _avg_outcomes(seed_runs)
    gate = E3Gate.evaluate(outcomes)
    return E3Result(arms={k: v.as_dict() for k, v in outcomes.items()},
                    gate=gate.as_dict())
