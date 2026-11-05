"""Non-RL baselines: Sequential SFT (1), SDFT (2a/2b), ICL few-shot judge (4),
Multi-Task static mixing (9).

These rows exist to answer "is RL even necessary?" and "is a trained judge even
necessary?".  They deliberately do **not** use the GRPO path, which is why the
trainer grew a ``step_supervised`` counterpart.  All of them consume L1 data
only (stage-initial demos), so the supervision ledger's gold ban applies
unchanged — a supervised baseline must not become an oracle by accident.

Row summaries from the plan:

=====  =============================================================  =======
Row    method                                                          key
=====  =============================================================  =======
1      Sequential SFT on the 150 demos of each stage                   ``seq_sft``
2a     SDFT: self-distillation on demos only                           ``sdft_only``
2b     SDFT-ICL: distill the model's own ICL prediction on *unlabelled* ``sdft_icl``
       prompts, so the number of prompts seen matches GRPO exactly
4      ICL few-shot judge (k=6 demos in the prompt), actor isolated    ``icl_judge``
9      Multi-task static mixing: all four stages jointly, no order     ``static_mix``
=====  =============================================================  =======
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..config.schema import Config
from ..methods.lora import AdapterSpec
from .base import BaseMethod, MethodPlan


@dataclass
class SequentialSFTMethod(BaseMethod):
    """Row 1: off-policy supervised fine-tuning, the classic forgetting lower bound."""

    name: str = "seq_sft"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    steps_per_stage: int = 300

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=False,
            supervised_only=True,
            notes=("off-policy SFT on 150 demos/stage; no reward signal, no judge",
                   "expected to forget severely — the lower bound in the run matrix"),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "SequentialSFTMethod":
        return cls(spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                                 alpha=float(cfg.model.lora_alpha)),
                   steps_per_stage=cfg.train.max_steps_per_stage)


@dataclass
class SDFTMethod(BaseMethod):
    """Row 2a/2b: self-distillation.

    ``icl=True`` (Row 2b) distils the model's own prediction made *with* a
    retrieved demonstration context, on every unlabelled prompt — matching
    GRPO's prompt budget.  ``icl=False`` (Row 2a) distils on the demos alone.
    """

    name: str = "sdft_only"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    icl: bool = False
    n_shots: int = 6
    temperature: float = 1.0
    steps_per_stage: int = 300

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=False,
            supervised_only=True,
            notes=(
                ("SDFT-ICL: distil own ICL prediction on unlabelled prompts "
                 f"({self.n_shots}-shot), aligned to GRPO's prompt budget")
                if self.icl else
                "SDFT: distil own prediction on the 150 demos only",
                "no external reward; isolates how much of the gain is RL rather than self-training",
            ),
        )

    @classmethod
    def from_config(cls, cfg: Config, icl: bool = False) -> "SDFTMethod":
        return cls(name="sdft_icl" if icl else "sdft_only",
                   spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                                 alpha=float(cfg.model.lora_alpha)),
                   icl=icl, n_shots=min(6, cfg.data.demo_per_stage),
                   steps_per_stage=cfg.train.max_steps_per_stage)


@dataclass
class ICLJudgeMethod(BaseMethod):
    """Row 4: judge is never trained; k demonstrations are put in the prompt."""

    name: str = "icl_judge"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    k_shots: int = 6

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=False,
            icl_judge=True,
            icl_shots=self.k_shots,
            notes=(f"in-context {self.k_shots}-shot judge; theta_J never updated",
                   "tests whether an 80-step warm-up is necessary at all"),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "ICLJudgeMethod":
        return cls(spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                                 alpha=float(cfg.model.lora_alpha)),
                   k_shots=6)


@dataclass
class StaticMixingMethod(BaseMethod):
    """Row 9: all stages jointly, order removed — the non-continual reference."""

    name: str = "static_mix"
    spec: AdapterSpec = field(default_factory=AdapterSpec.dual_adapter)
    stages: tuple[int, ...] = (1, 2, 3, 4)

    def plan(self) -> MethodPlan:
        return MethodPlan(
            name=self.name, spec=self.spec,
            judge_trainable_in_stage=False,
            static_mix=True,
            notes=("static mixture of all four stages (no ordering)",
                   "the reference point that defines the Continual Gap"),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "StaticMixingMethod":
        return cls(spec=AdapterSpec.dual_adapter(r=cfg.model.r_actor,
                                                 alpha=float(cfg.model.lora_alpha)),
                   stages=tuple(range(1, len(cfg.data.stages) + 1)))


# ---------------------------------------------------------------------------
# helpers used by the supervised arms
# ---------------------------------------------------------------------------


def demonstration_batch(world, stage: int, n: int, *, seed: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(prompt_ids, target_ids)`` for an off-policy demo batch.

    Targets are the demonstration answers for the stage.  This is L1
    supervision and it is *given* rather than verified — ``demo_answer`` is a
    separate entry point from ``correct_token`` precisely so that a real
    backend's verifier cannot be reached from here.
    """
    ids, _ = world.make_prompts(stage, n, seed=seed)
    targets = world.demo_answer(stage, ids)
    return ids, targets


def icl_context_prompts(world, stage: int, prompt_ids: torch.Tensor,
                        n_shots: int, *, seed: int) -> torch.Tensor:
    """Pack ``n_shots`` demonstrations ahead of each query prompt.

    The toy world has a fixed prompt length, so the context is a *set* of demo
    prompt ids concatenated and truncated back to the model's window: the point
    is that the query is conditioned on demonstrations, which is what Row 2b and
    Row 4 both rely on.
    """
    demo_ids, _ = world.make_prompts(stage, n_shots, seed=seed + 999)
    flat = torch.cat([demo_ids.reshape(-1), prompt_ids.reshape(-1)])
    L = world.prompt_len
    keep = (flat.numel() // L) * L
    return flat[:keep].reshape(-1, L)


def self_distillation_targets(model, prompt_ids: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
    """Soft targets from the model's own current distribution."""
    with torch.no_grad():
        logits = model.actor_logits(prompt_ids) / max(temperature, 1e-6)
        return F.softmax(logits, dim=-1)


def icl_judge_prompt(demos: Sequence[tuple[str, str, str]], answer: str,
                     n_shots: int = 6) -> str:
    """Build the in-context judge prompt used by Row 4.

    Each demonstration is ``(question, demo_answer, verdict)``.  Nothing is
    trained: the whole adaptation is the in-context prefix, which is exactly
    what distinguishes Row 4 from the warm-up baselines.
    """
    lines = ["Judge whether the candidate answer is correct.\n"]
    for q, a, v in list(demos)[:n_shots]:
        lines.append(f"Q: {q}\nA: {a}\nVerdict: {v}\n")
    lines.append(f"Q: the current question\nA: {answer}\nVerdict:")
    return "\n".join(lines)
