# csr_grpo — Continual Self-Rewarding GRPO testbed

A modular, supervision-audited implementation of the mechanism/diagnostic
protocol in `discovery/1-priority-analysis.html`: does task switching corrupt
self-reward, and does the corruption reach the policy gradient?

The defining constraint is **zero training-time gold access**. Every read of an
L3 verifier is gated by a runtime ledger, and a violation is an exception, not a
warning.

---

## What is here

```
csr_grpo/
  config/       dataclass schema + YAML loader + strict validation
  supervision/  L1/L2/L3 ledger (runtime gold ban), gold verifier adapters
  rewards/      GRPO / GOPO rank-only / degenerate-group zeroing; E2 reward arms
  diagnostics/  Acc_rank · Sign_agree · ρ_Adv · pseudo-contrast, 2×2 factorial,
                paired cluster bootstrap, matched-FPR early warning, dual line,
                ΔÂ_canon canonicalisation probe (H7, conditional)
  training/     bit-exact state snapshots (params + Adam + RNG), GRPO step,
                stage runner
  methods/      Dual-Adapter, Replay, RLER-faithful, judge ensemble, Fisher /
                stratified-random masking, LoRA primitives
                + Row 5b Meta-Rewarding, Row 5c online judge co-training,
                  Row 6 M-GRPO-adapted (EMA judge + entropy filter)
  experiments/  E1 (switch-vs-continue + gold DID + E1′), E2 (protocol A/B),
                E3 (method gate)
  backends/     toy CPU model (used here), HF backend (cluster)
tests/          86 tests, analytic rather than smoke
configs/        pre-registered presets
scripts/        cybertron submission helper, executable framework audit
```

## The supervision contract

| level | contents | training-time |
|-------|----------|---------------|
| L1 | stage demos, rule-perturbed negatives | allowed |
| L2 | self-consistency quorum, self-generated-input execution, `P_ext` Fisher labels | allowed |
| L3 | exact solvers, hidden tests, SymPy, IFEval/BFCL checkers | **forbidden** |

`Runner.train_step` opens `ledger.training_scope()`. Any L3 read inside it
raises `SupervisionViolation` before a verifier runs. Three roles may read L3
*outside* a training scope — `AUDIT_REPORTING`, `NI_TEST`,
`LOSS_GRADIENT_AUDIT` — plus the two declared diagnostic arms
(`E1_GOLD_DID_CONTROL`, `E2_DIAGNOSTIC_GOLD_ARM`).

Gold-control arms (E1's DID control, E2's gold reward, Row 10) legitimately
train on L3 rewards. They call `training_scope(gold_control=True)`, and the
ledger reports two separate counters:

```
gold_calls_in_training    = 0      # every trained method
gold_calls_in_gold_control = N     # declared reference arms only
```

`ledger.assert_clean()` enforces both.

## Quick start

```bash
python -m csr_grpo.cli plan   --config configs/dual_adapter_replay.yaml
python -m csr_grpo.cli selftest
python -m csr_grpo.cli e1     --config configs/e1_switch.yaml --steps 60 --heldout
python -m csr_grpo.cli e2     --config configs/dual_adapter_replay.yaml --stage 2
```

Every command writes `config.resolved.json`, `ledger.jsonl` and `result.json`
into its run directory, so any reported number can be traced to a configuration
and a gold-read count.

## The metrics

For a judge score matrix `S` and gold matrix `G` of shape `(P, G)`:

- **`Acc_rank`** — within-group pairwise accuracy over gold-discordant pairs.
  Degenerate groups (all correct / all wrong) have no denominator and are
  excluded, never filled with zero.
- **`Sign_agree`** — `P(sign Â_J == sign Â_gold)` on informative mixed groups.
- **`ρ_Adv`** — per-group Pearson of standardised advantages, aggregated via
  Fisher z. `MSE = 2(1−ρ)` holds identically; the tests assert it.
- **pseudo-contrast rate** — on gold-degenerate groups, the fraction where the
  judge still manufactures `max r − min r > τ`.

The 2×2 snapshot table decomposes exactly:

```
E11 − E00 = (E10 − E00) + (E01 − E00) + I_int
```

with `E_w` prepended at the stage boundary. The identity is verified per prompt
(`interaction_identity_residual < 1e-9`).

## Statistics

- **Paired cluster bootstrap** — resamples *prompts*, recomputes every cell on
  the same resample, so contrasts stay paired.
- **Non-inferiority** — one-sided paired test of `H0: Δ ≤ −δ_NI` at 2.5 pt,
  cluster-robust SE.
- **Three-way triage** — `go` (CI excludes 0) vs `equivalence` (CI inside
  ±δ_SESOI) vs `inconclusive` (neither). A resource stop is not a scientific
  claim, and the code cannot conflate them.
- **Early warning** — thresholds calibrated on the Continue arm to a target FPR
  (matched-FPR), then `Δt_lead` and leave-one-group-out `ΔR²_oos`.

## E2 protocols

**A (single step)** — freeze actor, rollout and Adam state; per reward source
restore, score the same rollouts, take one real optimiser step, read
`Δθ = θ_after − θ_before`. Reports `CosSim(Δθ, Δθ^gold)`,
`ΔL_old^(1st) = ∇L_old(θ)ᵀΔθ` and `‖P_{U_old}Δθ‖/‖Δθ‖`, where `U_old` comes
from an SVD of *per-prompt* old-task gradients.

**B (K steps)** — each branch re-samples from its own updated policy every step
and reports paired `ΔAcc_old` / `ΔAcc_new`.

## Method coverage

| Row | method key | mechanism | status |
|-----|-----------|-----------|--------|
| 3 | `shared_grpo` | shared adapter, judge updated in-stage | implemented |
| 5a | `rler_faithful` | M=2 sub-policies, matched rollout budget | implemented |
| 5a′ | `judge_lora_ensemble` | K=3 judge adapters (our ablation) | implemented |
| 5b | `meta_rewarding` | Meta-Judge → Judge preference training | implemented |
| 5c | `online_judge_cotraining` | per-step judge gradient + replay (**our adaptation**) | implemented |
| 6 | `mgrpo_adapted` | EMA judge τ=0.995 + entropy filter (**our adaptation**) | implemented |
| 7a/7b | `fisher_mask` / `random_mask` | fixed-arch masked vs stratified-random | implemented |
| 8 | `dual_adapter_replay` | r_S=0 + cross-stage replay | implemented |
| 10 | `gold_reference` | gold reward, reference only | implemented |
| — | `gopo_rank_only` | rank-only advantage (ablation f) | implemented |
| 1 | `seq_sft` | off-policy supervised fine-tuning on demos | implemented |
| 2a | `sdft_only` | self-distillation on demos (temperature-scaled teacher) | implemented |
| 2b | `sdft_icl` | self-distillation from an ICL-conditioned context | implemented |
| 4 | `icl_judge` | frozen judge, 6-shot in-context prefix | implemented |
| 9 | `static_mix` | all four stages in every batch (no ordering) | implemented |

Supervised rows take `train.sft_lr` (default `1e-2`), not `train.lr`: a
cross-entropy update on demonstrations needs a step size roughly three orders of
magnitude larger than a policy-gradient step. Reusing `train.lr` leaves the
model frozen — measured on the toy backend, lr=1e-5 moved demo accuracy
0.141 → 0.141 over 30 steps, while 1e-2 reached 1.000 in 100.

## Known limitations of the toy backend

The CPU toy world is a testbed, not a result. It preserves the causal structure
the metrics need (four heterogeneous stages, a judge whose adapter drifts, a
shared-vs-disjoint architecture that makes contamination real), but it is small
enough that effect sizes are noise. Two limits are recorded honestly in
`discovery/2-problem.html`:

1. the judge's correlation with gold is weak, so `Sign_agree` sits near chance;
2. LoRA's zero-init means `B` moves before `A` does, so short runs show small
   `Δθ` norms.

Neither affects the correctness of the estimators, which are tested against
closed-form values — but no *scientific* claim should be read from toy runs.

## Framework audit

```bash
python scripts/audit_framework.py --with-pytest
```

Verifies deliverable traceability, metric semantics against closed-form values,
the statistical contracts (pairing, matched FPR, non-inferiority direction,
three-way triage), and the supervision invariant. Exits non-zero on any failure
and prints JSON so the report can be attached to a run manifest.

## Running on the cluster

```bash
./scripts/submit_cybertron.sh --project <p> --cluster <c> --pool <pool> \
    --image <img> --gpu 8 --command "python -m csr_grpo.cli e1 ..."
```

See `scripts/submit_cybertron.sh --help`.
