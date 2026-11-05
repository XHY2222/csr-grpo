"""E1 — Switch vs. Continue with a Gold double-difference control (H1/C1).

Design (paper §E1)
------------------
From one Stage-1 checkpoint, fork two arms with **matched update budget**:

* ``continue`` — keep self-reward GRPO on *fresh* Stage-1 prompts;
* ``switch``   — self-reward GRPO on Stage 2 prompts.

Both are measured on the *old* Stage-1 ``P_audit`` line (and on their own
on-policy line).  The raw switch increment is::

    Δ_switch = (metric_switch − metric_continue)

which is contaminated by ordinary continual-learning interference.  To isolate
the self-reward-specific part we repeat both arms with **gold rewards**
(``reward_source=GOLD``, a declared gold-control arm) and take::

    Δ_DID = Δ_switch^self − Δ_switch^gold

Everything is measured per prompt, so the paired cluster bootstrap can put a CI
on both ``Δ_switch`` and ``Δ_DID``.  The pre-registered decision rule is in
``diagnostics.bootstrap.classify_effect``: the CI containing 0 is a *resource*
stop, while a CI fully inside ±δ_SESOI is a *scientific* equivalence claim.

E1′ is the same routine run on the held-out ``Stage 2 → 3`` transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..compat import to_np
from ..config.schema import Config
from ..diagnostics.bootstrap import (
    BootstrapCI,
    Decision,
    classify_effect,
    paired_cluster_bootstrap,
)
from ..diagnostics.dual_line import LINE_AUDIT, LINE_ONPOLICY
from ..diagnostics.fidelity import PerPromptStats, per_prompt_stats
from ..methods.base import make_method
from ..rewards.sources import RewardSource
from ..supervision.ledger import Role, SupervisionLedger
from ..training.runner import Runner, StageContext


@dataclass
class ArmResult:
    name: str
    reward_source: str
    stage: int
    final_stats: dict[str, PerPromptStats]     # per line
    series: dict[str, dict[str, list[float]]]
    steps: list[int]

    def metric_on(self, line: str, metric: str = "acc_rank") -> PerPromptStats:
        return self.final_stats[line]

    def as_row(self) -> dict:
        out = {"arm": self.name, "reward": self.reward_source, "stage": self.stage}
        for line, series in self.series.items():
            for metric, vals in series.items():
                out[f"{line}.{metric}"] = vals[-1] if vals else float("nan")
        return out


def run_arm(cfg: Config, *, reward_source: RewardSource, target_stage: int,
            steps: int, seed: int, ledger: SupervisionLedger,
            audit_stage: int = 1, warmup_rounds: int = 16,
            measure_every: int = 20,
            audit_seed: int | None = None) -> ArmResult:
    """Run one E1 arm from a fresh Stage-1 state.

    ``audit_seed`` fixes the ``P_audit`` prompt set.  The Switch and Continue
    arms are contrasted *within* prompt, so they must see the same questions;
    leaving it ``None`` derives the seed from the target stage, which makes the
    two arms measure different prompts and silently turns the paired bootstrap
    into an unpaired one.
    """
    # deep copy: E1 calls this four times and shallow copies shared the
    # nested dataclasses, so one arm's reward source leaked into the next
    import copy

    cfg = copy.deepcopy(cfg)
    cfg.reward_source = reward_source.value
    cfg.is_diagnostic_arm = reward_source is RewardSource.GOLD
    cfg.seed = seed
    cfg.stages_to_run = (audit_stage, target_stage)
    cfg.validate()

    method = make_method(cfg)
    runner = Runner(cfg, method.plan(), ledger)

    # Stage-1 warm-up so both arms start from a comparable judge
    runner.warmup_judge(audit_stage, n_rounds=warmup_rounds)
    runner.freeze_judge()

    ctx = runner.make_context(target_stage, n_prompts=cfg.data.n_prompts_per_stage,
                              seed=1000 + target_stage, audit_stage=audit_stage,
                              audit_seed=audit_seed)
    # a frozen answer set on P_audit keeps the two arms' measurement identical
    n_audit = len(ctx.audit_prompts)
    G = cfg.train.group_size
    runner.model.set_stage(audit_stage)
    with torch.no_grad():
        seqs, _, _ = runner.model.sample(ctx.audit_prompts, G, generator=runner.rng)
    audit_answers = seqs[:, -1].reshape(n_audit, G)

    series: dict[str, dict[str, list[float]]] = {}
    measured_steps: list[int] = []
    last_obs: dict[str, object] = {}
    for step in range(1, steps + 1):
        runner.train_step(ctx, step)
        if step % measure_every == 0 or step == steps:
            obs_a = runner._measure_audit_fixed(ctx, step, audit_answers, audit_stage)
            obs_b = runner.measure_onpolicy(ctx, step)
            measured_steps.append(step)
            for obs in (obs_a, obs_b):
                bucket = series.setdefault(obs.line, {})
                for k, v in obs.values.items():
                    bucket.setdefault(k, []).append(v)
                last_obs[obs.line] = obs
            if runner.judge_optimizer is not None:
                runner.warmup_judge(target_stage, n_rounds=2)

    # final per-prompt stats on the audit line (for paired bootstrap)
    judge = to_np(runner.model.judge_score_matrix(ctx.audit_prompts, audit_answers))
    gold = to_np(runner.score_with_gold(audit_stage, ctx.audit_prompts,
                                 audit_answers))
    final = {
        LINE_AUDIT: per_prompt_stats(judge, gold, tau=cfg.train.degenerate_tau),
    }
    onp = runner.dual_line.latest(LINE_ONPOLICY)
    if onp is not None:
        final[LINE_ONPOLICY] = onp.stats

    return ArmResult(name=f"{reward_source.value}@{target_stage}",
                     reward_source=reward_source.value, stage=target_stage,
                     final_stats=final, series=series, steps=measured_steps)


@dataclass
class E1Result:
    delta_switch_self: BootstrapCI
    delta_switch_gold: BootstrapCI | None
    delta_did: BootstrapCI | None
    decision_switch: Decision
    decision_did: Decision
    arms: list[dict] = field(default_factory=list)
    metric: str = "sign_agree"
    line: str = LINE_AUDIT
    # H1's second half: matched-FPR lead time, when both arms recorded a series
    lag: dict | None = None

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "line": self.line,
            "structural_lag": self.lag,
            "delta_switch_self": self.delta_switch_self.as_dict(),
            "delta_switch_gold": self.delta_switch_gold.as_dict()
            if self.delta_switch_gold else None,
            "delta_did": self.delta_did.as_dict() if self.delta_did else None,
            "decision_switch": self.decision_switch,
            "decision_did": self.decision_did,
            "arms": self.arms,
        }


def lead_time_from_arms(
    switch_arm: ArmResult, continue_arm: ArmResult, *,
    line: str = LINE_AUDIT, target_fpr: float = 0.05, min_required: int = 50,
    early_metric: str = "sign_agree", late_metric: str = "acc_rank",
    forward_line: str = LINE_ONPOLICY, n_resamples: int = 2000, seed: int = 0,
) -> dict | None:
    """``Δt_lead`` under matched FPR **and** ``ΔR²_oos``, the two pre-registered
    structural-lag criteria (paper §H1).

    Criterion 1 (``Δt_lead``): matched-FPR lead time, calibrated on the Continue
    arm.  Criterion 2 (``ΔR²_oos``): leave-one-step-out cross-step regression
    predicting the forward metric change ``ΔFM`` from the audit-line metrics,
    comparing ``acc_rank`` alone vs ``acc_rank + sign_agree``.  Either criterion
    being met supports H1's structural-lag claim.

    Returns ``None`` when either arm recorded no series (e.g. a very short run).
    ``ΔR²_oos`` is reported as ``None`` when there are too few steps for a
    meaningful regression (minimum 5 consecutive measurements).
    """
    from ..diagnostics.early_warning import (
        LagVerdict, matched_fpr_lead_time, predictive_gain,
    )

    sw = switch_arm.series.get(line, {})
    co = continue_arm.series.get(line, {})
    if not sw or not co:
        return None
    if early_metric not in sw or late_metric not in sw:
        return None
    if early_metric not in co or late_metric not in co:
        return None
    steps = list(switch_arm.steps)
    n = min(len(sw[early_metric]), len(co[early_metric]), len(steps))
    if n < 3:
        return None

    lead = matched_fpr_lead_time(
        {early_metric: [sw[early_metric][:n]], late_metric: [sw[late_metric][:n]]},
        {early_metric: [co[early_metric][:n]], late_metric: [co[late_metric][:n]]},
        steps[:n], target_fpr=target_fpr, min_required=min_required,
        early_metric=early_metric, late_metric=late_metric,
    )

    # Criterion 2: ΔR²_oos.  The target is ΔFM — the step-to-step change in the
    # forward metric (on-policy acc_rank).  The base predictor is the audit-line
    # acc_rank; the extra predictor is the audit-line sign_agree.  If adding
    # sign_agree predicts ΔFM better out-of-sample, the early metric carries
    # information the late one does not — which is the structural claim.
    r2 = None
    fwd = switch_arm.series.get(forward_line, {})
    if late_metric in fwd and early_metric in sw and n >= 5:
        fm = np.array(fwd[late_metric][:n], dtype=float)
        if len(fm) >= 5 and not np.allclose(fm, fm[0]):
            target = np.diff(fm)                       # ΔFM_t = FM_{t+1} - FM_t
            base = np.array(sw[late_metric][:len(target)], dtype=float)
            extra = np.array(sw[early_metric][:len(target)], dtype=float)
            groups = np.arange(len(target))            # leave-one-step-out
            if len(target) >= 5 and np.std(target) > 1e-12:
                r2 = predictive_gain(base, extra, target, groups,
                                     n_resamples=min(n_resamples, 500),
                                     seed=seed)

    verdict = LagVerdict(lead=lead, r2=r2)
    return verdict.as_dict()


def compare_arms(
    self_arm_a: ArmResult, self_arm_b: ArmResult,
    gold_arm_a: ArmResult | None = None, gold_arm_b: ArmResult | None = None,
    *, metric: str = "sign_agree", line: str = LINE_AUDIT,
    sesoi_pt: float = 2.0, n_resamples: int = 2000, seed: int = 0,
    level: float = 0.95,
) -> E1Result:
    """Paired bootstrap on Δ_switch and Δ_DID.

    ``a`` is the Switch arm and ``b`` the Continue arm for each reward source.
    """
    cells = {"switch_self": self_arm_a.final_stats[line],
             "continue_self": self_arm_b.final_stats[line]}
    contrasts = [("switch_self", "continue_self")]
    if gold_arm_a is not None and gold_arm_b is not None:
        cells["switch_gold"] = gold_arm_a.final_stats[line]
        cells["continue_gold"] = gold_arm_b.final_stats[line]
        contrasts.append(("switch_gold", "continue_gold"))

    res = paired_cluster_bootstrap(cells, metric, contrasts,
                                   n_resamples=n_resamples, seed=seed,
                                   level=level)
    d_self = res.contrasts["switch_self-continue_self"]
    d_gold = res.contrasts.get("switch_gold-continue_gold")
    did = None
    decision_did: Decision = "inconclusive"
    if d_gold is not None:
        samples = d_self.boot - d_gold.boot
        from ..diagnostics.bootstrap import percentile_ci

        did = percentile_ci(samples, d_self.point - d_gold.point)
        decision_did = classify_effect(did, sesoi_pt=sesoi_pt)

    return E1Result(
        delta_switch_self=d_self,
        delta_switch_gold=d_gold,
        delta_did=did,
        decision_switch=classify_effect(d_self, sesoi_pt=sesoi_pt),
        decision_did=decision_did,
        arms=[self_arm_a.as_row(), self_arm_b.as_row()],
        metric=metric,
        line=line,
    )


def _merge_arms(arms: list[ArmResult]) -> ArmResult:
    """Pool several seed-replicate arms into one by concatenating per-prompt
    stats on each line.  The series (step-indexed trajectories) are taken from
    the first arm, since every seed sees the same measurement schedule."""
    if len(arms) == 1:
        return arms[0]
    from ..diagnostics.fidelity import PerPromptStats

    merged_stats: dict[str, PerPromptStats] = {}
    for line in arms[0].final_stats:
        merged_stats[line] = PerPromptStats.concat(
            [a.final_stats[line] for a in arms if line in a.final_stats])
    return ArmResult(
        name=arms[0].name,
        reward_source=arms[0].reward_source,
        stage=arms[0].stage,
        final_stats=merged_stats,
        series=arms[0].series,
        steps=arms[0].steps,
    )


def run_e1(cfg: Config, ledger: SupervisionLedger, *, seeds: int = 1,
           steps: int | None = None, gold_seeds: int = 1,
           target_stage: int = 2, audit_stage: int = 1,
           metric: str = "sign_agree", measure_every: int = 20) -> E1Result:
    """Full E1: self-reward Switch/Continue + gold-reward controls + DID."""
    steps = steps or min(cfg.train.max_steps_per_stage, 80)

    # One P_audit prompt set for the whole experiment: the Switch and Continue
    # arms are compared per prompt, and the gold arms are the DID control, so
    # all four must be measured on the same questions.  The base seed is
    # configurable (``cfg.exp.audit_seed``); offsetting by ``audit_stage``
    # makes E1 (audit_stage=1) and E1' (audit_stage=2) use disjoint P_audit
    # sets so the held-out confirmation is genuinely independent.
    audit_seed = cfg.exp.audit_seed + audit_stage

    def _pair(reward: RewardSource, seed: int):
        a = run_arm(cfg, reward_source=reward, target_stage=target_stage,
                    steps=steps, seed=seed, ledger=ledger, audit_stage=audit_stage,
                    audit_seed=audit_seed, measure_every=measure_every)
        b = run_arm(cfg, reward_source=reward, target_stage=audit_stage,
                    steps=steps, seed=seed + 100, ledger=ledger,
                    audit_stage=audit_stage, audit_seed=audit_seed,
                    measure_every=measure_every)
        return a, b

    self_arms_a, self_arms_b = [], []
    for s in range(seeds):
        a, b = _pair(RewardSource.SELF, s * 10)
        self_arms_a.append(a)
        self_arms_b.append(b)
    # combine across seeds by pooling the final per-prompt stats
    self_a = _merge_arms(self_arms_a)
    self_b = _merge_arms(self_arms_b)
    gold_a = gold_b = None
    if gold_seeds > 0:
        gold_arms_a, gold_arms_b = [], []
        for s in range(gold_seeds):
            ga, gb = _pair(RewardSource.GOLD, 700 + s * 10)
            gold_arms_a.append(ga)
            gold_arms_b.append(gb)
        gold_a = _merge_arms(gold_arms_a)
        gold_b = _merge_arms(gold_arms_b)
    result = compare_arms(self_a, self_b, gold_a, gold_b, metric=metric,
                          sesoi_pt=cfg.exp.delta_sesoi_pt,
                          n_resamples=cfg.exp.bootstrap_resamples,
                          level=cfg.exp.ci_level)
    result.lag = lead_time_from_arms(
        self_a, self_b, target_fpr=cfg.exp.matched_fpr,
        min_required=cfg.exp.dt_lead_steps)
    return result


# ---------------------------------------------------------------------------
# E1' -- held-out Stage 2 -> 3 confirmation
# ---------------------------------------------------------------------------


def run_e1_prime(cfg: Config, ledger: SupervisionLedger, *, steps: int = 60,
                 metric: str = "sign_agree", measure_every: int = 20) -> E1Result:
    """Same protocol on the held-out transition (Stage 2 → 3)."""
    return run_e1(cfg, ledger, steps=steps, gold_seeds=1,
                  target_stage=3, audit_stage=2, metric=metric,
                  measure_every=measure_every)
