"""Non-RL baselines: supervised step maths, ICL prompt, and ledger cleanliness."""

import numpy as np
import pytest
import torch

from csr_grpo.config.schema import Config
from csr_grpo.methods.base import make_method
from csr_grpo.methods.baselines import (
    demonstration_batch, icl_context_prompts, icl_judge_prompt,
    self_distillation_targets,
)
from csr_grpo.methods.lora import AdapterSpec
from csr_grpo.supervision.ledger import SupervisionLedger
from csr_grpo.training.runner import Runner


def _cfg(method="seq_sft"):
    cfg = Config()
    cfg.method = method
    cfg.model.r_actor = 16
    cfg.model.lora_alpha = 32
    cfg.train.group_size = 4
    cfg.train.warmup_rounds = 5
    cfg.data.n_prompts_per_stage = 16
    return cfg


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["seq_sft", "sdft_only", "sdft_icl",
                                  "icl_judge", "static_mix"])
def test_baseline_never_trains_a_judge(name):
    cfg = Config()
    cfg.method = name
    cfg.validate()
    plan = make_method(cfg).plan()
    assert plan.judge_trainable_in_stage is False
    assert plan.replay is False


def test_sdft_icl_and_only_are_distinguished():
    cfg = Config()
    cfg.method = "sdft_icl"
    icl = make_method(cfg).plan()
    cfg.method = "sdft_only"
    only = make_method(cfg).plan()
    assert "ICL" in " ".join(icl.notes)
    assert "ICL" not in " ".join(only.notes)


def test_static_mixing_covers_all_configured_stages():
    cfg = Config()
    cfg.method = "static_mix"
    method = make_method(cfg)
    assert method.stages == (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# supervised step
# ---------------------------------------------------------------------------


def test_sft_step_reduces_target_nll():
    """One SFT step must lower the NLL of the demonstrated answers."""
    led = SupervisionLedger()
    cfg = _cfg()
    runner = Runner(cfg, make_method(cfg).plan(), led)
    runner.model.set_stage(1)
    ids, targets = demonstration_batch(runner.world, 1, 8, seed=0)

    def nll():
        with torch.no_grad():
            logp = torch.log_softmax(runner.model.actor_logits(ids), dim=-1)
            return float(-logp.gather(1, targets[:, None]).mean())

    before = nll()
    for _ in range(5):
        runner.trainer.step_supervised(ids, targets)
    after = nll()
    assert after < before, f"SFT did not reduce NLL: {before:.4f} -> {after:.4f}"


def test_supervised_step_is_gold_free():
    led = SupervisionLedger()
    cfg = _cfg()
    runner = Runner(cfg, make_method(cfg).plan(), led)
    runner.model.set_stage(1)
    ids, targets = demonstration_batch(runner.world, 1, 4, seed=1)
    runner.trainer.step_supervised(ids, targets)
    s = led.summary()
    assert s["gold_calls_in_training"] == 0
    assert s["violations"] == 0


def test_supervised_step_cannot_read_l3_inside_scope():
    """A supervised step opens a training scope, so L3 stays blocked."""
    from csr_grpo.supervision.ledger import Level, Role, SupervisionViolation

    led = SupervisionLedger()
    with led.training_scope(), pytest.raises(SupervisionViolation):
        led.access(Level.L3, Role.AUDIT_REPORTING, "inside supervised step")


def test_sdft_kl_step_moves_toward_teacher():
    """A KL step must reduce the gap to a teacher that is *not* the student.

    Using the student itself as teacher would start at KL ≈ 0 and give the test
    nothing to measure, so the teacher is a sharpened (temperature-scaled)
    version of the current distribution.
    """
    led = SupervisionLedger()
    cfg = _cfg("sdft_only")
    runner = Runner(cfg, make_method(cfg).plan(), led)
    runner.model.set_stage(1)
    ids, _ = demonstration_batch(runner.world, 1, 4, seed=2)
    with torch.no_grad():
        teacher = torch.softmax(runner.model.actor_logits(ids) / 2.5, dim=-1)

    def kl():
        with torch.no_grad():
            logp = torch.log_softmax(runner.model.actor_logits(ids), dim=-1)
            return float(torch.sum(teacher * (teacher.log() - logp), dim=-1).mean())

    before = kl()
    assert before > 1e-4, "teacher must differ from the student for this test"
    for _ in range(5):
        runner.trainer.step_supervised(ids, torch.zeros(len(ids), dtype=torch.long),
                                       teacher_logprob=teacher)
    assert kl() < before


def test_teacher_targets_are_a_distribution():
    led = SupervisionLedger()
    cfg = _cfg()
    runner = Runner(cfg, make_method(cfg).plan(), led)
    runner.model.set_stage(1)
    ids, _ = demonstration_batch(runner.world, 1, 3, seed=3)
    p = self_distillation_targets(runner.model, ids)
    assert torch.allclose(p.sum(dim=-1), torch.ones(len(ids)), atol=1e-5)
    assert torch.all(p >= 0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_demonstration_batch_shapes():
    from csr_grpo.backends.toy import ToyWorld

    w = ToyWorld()
    ids, targets = demonstration_batch(w, 2, 5, seed=0)
    assert ids.shape == (5, w.prompt_len)
    assert targets.shape == (5,)
    assert torch.equal(targets, w.correct_token(2, ids))


def test_icl_context_is_divisible_by_prompt_len():
    from csr_grpo.backends.toy import ToyWorld

    w = ToyWorld()
    ids, _ = demonstration_batch(w, 1, 4, seed=0)
    ctx = icl_context_prompts(w, 1, ids, n_shots=3, seed=0)
    assert ctx.shape[1] == w.prompt_len
    assert ctx.shape[0] >= 4          # query prompts survive the truncation


def test_icl_judge_prompt_renders_demos_and_candidate():
    text = icl_judge_prompt([("q1", "a1", "Yes"), ("q2", "a2", "No")],
                            "candidate answer", n_shots=2)
    assert "Q: q1" in text and "Verdict: Yes" in text
    assert "A: a2" in text
    assert text.rstrip().endswith("Verdict:")


def test_icl_judge_prompt_respects_n_shots():
    demos = [(f"q{i}", f"a{i}", "Yes") for i in range(10)]
    text = icl_judge_prompt(demos, "cand", n_shots=3)
    assert text.count("Q: ") == 4     # 3 demos + the candidate
