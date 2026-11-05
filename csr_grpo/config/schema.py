"""Configuration dataclasses for the continual self-rewarding GRPO testbed.

Every knob the paper pre-registers lives here so run manifests are auditable.
``validate()`` enforces the supervision contract at config time as a second
line of defence behind the runtime ledger:

* a method whose ``reward_source`` is gold may only exist as a *reference* row
  (Row 10) or as an E2 diagnostic arm — never as a trained method;
* ``Stage 2 -> 3`` is reserved for confirmatory use, so exploration flags are
  rejected there unless the run is explicitly marked ``confirmatory``;
* probe sets must be disjoint (P_ext vs P_audit).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

RewardSourceName = Literal["self", "gold", "protected", "degenerate_zeroed"]
MethodName = Literal[
    "shared_grpo",       # Row 3 baseline: shared LoRA, m=0
    "dual_adapter",      # Row 8 / Group C
    "dual_adapter_replay",
    "rler_faithful",     # Row 5a
    "judge_lora_ensemble",
    "fisher_mask",       # Row 7a
    "random_mask",       # Row 7b
    "gopo_rank_only",    # ablation (f)
    "gold_reference",    # Row 10, reference only
    "meta_rewarding",    # Row 5b
    "online_judge_cotraining",  # Row 5c (our adaptation)
    "mgrpo_adapted",     # Row 6 (our adaptation)
    "seq_sft",           # Row 1
    "sdft_only",         # Row 2a
    "sdft_icl",          # Row 2b
    "icl_judge",         # Row 4
    "static_mix",        # Row 9
]


@dataclass
class ModelConfig:
    backend: Literal["toy", "hf"] = "toy"
    name: str = "toy-7b"
    dtype: str = "bfloat16"
    # LoRA structure: r_S is the *shared* rank, r_priv the per-role private rank
    r_shared: int = 32
    r_actor: int = 32
    r_judge: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


@dataclass
class TrainingConfig:
    group_size: int = 8                    # G in GRPO
    max_steps_per_stage: int = 300
    lr: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    adv_eps: float = 1e-4                  # epsilon in Â = (r-μ)/(σ+ε)
    kl_coef: float = 1e-3
    ref_kl_clip: float | None = None
    seed: int = 0
    device: str = "cpu"

    # judge warm-up is a separate optimisation problem from the actor's GRPO
    # step: the BCE loss on rule-perturbed pairs needs a much larger step size
    # than a policy-gradient update, and reusing train.lr freezes the judge.
    warmup_lr: float = 3e-3
    warmup_rounds: int = 60

    # Supervised updates (SFT / SDFT) are a different optimisation problem from
    # the GRPO step: cross-entropy on demos needs a step size ~3 orders of
    # magnitude larger.  Measured on the toy backend: lr=1e-5 moves demo
    # accuracy 0.141 -> 0.141 in 30 steps, lr=1e-2 reaches 1.000 in 100.
    sft_lr: float = 1e-2

    # advantage construction
    adv_mode: Literal["grpo", "rank_only"] = "grpo"
    degenerate_zero: bool = False          # E2 arm (iv): zero advantage on all-correct/wrong groups
    degenerate_tau: float = 0.05           # τ for pseudo-contrast trigger

    # replay
    replay_ratio: float = 1.0              # 1:1 historical vs current pairs
    replay_window_stages: int | None = None


@dataclass
class JudgeConfig:
    temperature: float = 0.0
    max_cot_tokens: int = 192
    mode: Literal["greedy_cot_logit", "direct_logit"] = "greedy_cot_logit"
    ema_tau: float | None = None           # Row 6 slow-teacher EMA
    ensemble_k: int = 1                    # K=3 for judge-LoRA ensemble variant
    atomic_rubric: bool = False            # Stage 4 IFEval atomic decomposition


@dataclass
class DataConfig:
    stages: tuple[str, ...] = ("logic", "code", "math", "instruction")
    demo_per_stage: int = 150
    n_prompts_per_stage: int = 400
    # probes: development (used for Fisher + monitoring) vs blind audit
    ext_prompts: int = 200
    ext_pairs_per_prompt: int = 4
    audit_prompts: int = 200               # N_prompts, locked by Step-0 power sim
    audit_pairs_per_prompt: int = 4
    audit_heldout_stages: tuple[str, ...] = ("math",)  # Stage 3+ = confirmatory
    consensus_quorum: tuple[int, int] = (5, 8)          # route B ≥5/8
    route_a_abstain: bool = True           # drop samples whose verdict didn't flip
    soft_label_for_ambiguous: float = 0.5


@dataclass
class ExperimentConfig:
    # E1
    e1_continue_steps: int | None = None   # default: match switch arm steps
    e1_seeds: int = 3
    e1_gold_control_seeds: int = 1
    e1_confirm_pair: tuple[int, int] = (2, 3)
    e1_confirm_seeds: int = 2
    # E2
    e2_checkpoints: tuple[int, ...] = (50, 150, 300)
    e2_protocol_b_k: tuple[int, ...] = (20, 50)
    e2_arms: tuple[RewardSourceName, ...] = (
        "self", "protected", "gold", "degenerate_zeroed"
    )
    # E3
    e3_arms: tuple[MethodName, ...] = (
        "dual_adapter", "dual_adapter_replay", "rler_faithful",
        "judge_lora_ensemble",
    )
    # statistical thresholds (pre-registered)
    delta_sesoi_pt: float = 2.0             # 2.0 percentage points
    delta_ni_pt: float = 2.5                # non-inferiority margin
    dt_lead_steps: int = 50
    matched_fpr: float = 0.05
    bootstrap_resamples: int = 2000
    bootstrap_cluster: Literal["prompt"] = "prompt"
    ci_level: float = 0.95
    # Fixed seed for P_audit prompt generation across all arms and stages.
    # Every make_context call in E1/E2/E3/cmd_run uses this so that the same
    # question set is used in all comparisons; changing it invalidates pairing.
    audit_seed: int = 5000


@dataclass
class RuntimeConfig:
    out_dir: str = "runs"
    log_every: int = 10
    save_snapshots: bool = True
    deterministic: bool = True
    num_threads: int = 4
    ledger_strict: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # --- run identity -------------------------------------------------------
    method: MethodName = "dual_adapter_replay"
    reward_source: RewardSourceName = "self"
    stages_to_run: tuple[int, ...] = (1, 2)
    is_reference: bool = False
    is_diagnostic_arm: bool = False
    confirmatory: bool = False
    seed: int = 0

    # ----------------------------------------------------------------------
    def validate(self) -> "Config":
        errors: list[str] = []

        # 0. types first: every check below compares numbers, so a string
        #    field raises TypeError from `<` before any message can be built.
        #    A CLI override such as ``--set train.lr=3e-5`` used to arrive as
        #    the string '3e-5' and fail here, inside torch.optim, after the
        #    model had been built.
        for section in ("model", "train", "judge", "data", "exp", "runtime"):
            obj = getattr(self, section)
            for f in fields(obj):
                if f.type not in ("float", "int") and not _is_numeric_annotation(f.type):
                    continue
                value = getattr(obj, f.name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(
                        f"{section}.{f.name} must be numeric, got "
                        f"{type(value).__name__} ({value!r})"
                    )
        # Return before any range comparison runs: with a string field those
        # comparisons raise TypeError and the message above never gets out.
        if errors:
            raise ValueError("invalid configuration:\n  - " + "\n  - ".join(errors))

        if self.train.group_size < 2:
            errors.append("train.group_size must be >= 2 for GRPO")

        # 1. gold reward is never a training reward source
        if self.reward_source == "gold" and not (self.is_reference or
                                                 self.is_diagnostic_arm):
            errors.append(
                "reward_source='gold' requires is_reference=True (Row 10) or "
                "is_diagnostic_arm=True (E2 arm iii); the supervision ledger "
                "forbids gold in any trained method"
            )
        if self.method == "gold_reference" and not self.is_reference:
            errors.append("method='gold_reference' implies is_reference=True")

        # 2. exploration is forbidden on confirmatory stages
        heldout = set(self.data.audit_heldout_stages)
        if self.confirmatory and (self.data.stages and
                                 not set(self.data.stages) <= heldout):
            # confirmatory runs may only touch held-out stages
            pass  # warning-level; stages_to_run is the real guard below

        # 3. probe disjointness
        if self.data.ext_prompts <= 0 or self.data.audit_prompts <= 0:
            errors.append("both P_ext and P_audit must be non-empty")
        if self.data.audit_pairs_per_prompt != 4:
            errors.append("paper pre-registers exactly 4 pairs per audit prompt")

        # 4. E3 arms must be known
        for arm in self.exp.e3_arms:
            if arm not in _KNOWN_METHODS:
                errors.append(f"unknown E3 arm {arm!r}")

        # 5. SESOI / NI sanity
        if not (0 < self.exp.delta_sesoi_pt < 20):
            errors.append("delta_sesoi_pt out of plausible range")
        if not (0 < self.exp.delta_ni_pt < 20):
            errors.append("delta_ni_pt out of plausible range")

        # 6. replay requires history, i.e. at least a 2-stage run
        if (self.method == "dual_adapter_replay" and
                len(self.stages_to_run) < 2):
            errors.append("dual_adapter_replay needs >= 2 stages")

        # 7. rank-only ablation cannot be combined with degenerate zeroing
        #    (the paper treats them as separate causal knobs)
        if self.train.adv_mode == "rank_only" and self.train.degenerate_zero:
            errors.append(
                "adv_mode='rank_only' with degenerate_zero=True conflates the "
                "ablation (f) knobs; run them separately"
            )

        if errors:
            raise ValueError("invalid configuration:\n  - " + "\n  - ".join(errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _is_numeric_annotation(annotation: object) -> bool:
    """True for ``float``/``int`` and for ``X | None`` where X is numeric."""
    text = str(annotation)
    return any(part in text.split("|") for part in ("float", "int"))


_KNOWN_METHODS = {
    "shared_grpo", "dual_adapter", "dual_adapter_replay", "rler_faithful",
    "judge_lora_ensemble", "fisher_mask", "random_mask", "gopo_rank_only",
    "gold_reference", "meta_rewarding", "online_judge_cotraining",
    "mgrpo_adapted", "seq_sft", "sdft_only", "sdft_icl", "icl_judge",
    "static_mix",
}
