"""Backend protocol conformance (`backends/protocol.py`).

The protocol exists so an incomplete backend fails at construction rather than
on the first training step, after warm-up and model loading.  These tests pin
that behaviour: the error must name every missing member, and both shipped
backends must satisfy the contract.
"""

import pytest
import torch

from csr_grpo.backends.protocol import (
    REQUIRED_MODEL_METHODS,
    BackendProtocolError,
    check_model,
    check_world,
    describe,
)
from csr_grpo.backends.toy import ToyModel, ToyWorld
from csr_grpo.methods.lora import AdapterSpec


def _toy_model():
    return ToyModel(ToyWorld(), AdapterSpec.dual_adapter(r=8), seed=0)


def test_toy_model_satisfies_the_model_protocol():
    check_model(_toy_model())
    report = describe(_toy_model())
    assert report["absent"] == []
    assert report["judge_bank"] is True
    assert report["complete"] is True
    # judge_bank is set in __init__, so a class-level report cannot see it --
    # it must say "unknown", never "missing".
    assert describe(ToyModel)["judge_bank"] == "unknown"


def test_toy_world_satisfies_the_world_protocol():
    check_world(ToyWorld())


def test_hf_backend_satisfies_the_model_protocol():
    """The HF backend must expose every method the runner calls, independent of
    whether weights are present — the protocol is about the interface."""
    from csr_grpo.backends.hf import HFBackend

    assert describe(HFBackend)["absent"] == []
    # judge_bank is a property on the class, so this one resolves either way
    assert describe(HFBackend)["judge_bank"] == "present"


def test_protocol_does_not_require_backend_internal_switching():
    """`set_view` is PEFT's mechanism, not a training-loop requirement.  If it
    were required, every backend would have to adopt PEFT's global-state model
    of adapter activation."""
    assert "set_view" not in REQUIRED_MODEL_METHODS
    assert "view_parameters" in REQUIRED_MODEL_METHODS


def test_missing_method_is_reported_by_name():
    class Partial:
        view_parameters = named_parameters = parameters = set_stage = None

        def actor_logits(self, *a, **k):
            ...

    with pytest.raises(BackendProtocolError) as exc:
        check_model(Partial(), name="Partial")

    msg = str(exc.value)
    assert "Partial" in msg
    # every missing member is named, not just the first
    for missing in ("judge_score_matrix", "gold_score_matrix", "sample", "logprob"):
        assert missing in msg
    assert "first training step" in msg


def test_missing_judge_bank_is_reported():
    class NoBank:
        view_parameters = named_parameters = parameters = set_stage = None
        actor_logits = judge_score_matrix = judge_prob = None
        gold_score_matrix = sample = logprob = None

    with pytest.raises(BackendProtocolError, match="judge_bank"):
        check_model(NoBank())


def test_non_callable_attribute_does_not_count_as_implemented():
    class Wrong:
        view_parameters = "not callable"
        named_parameters = parameters = set_stage = None
        actor_logits = judge_score_matrix = judge_prob = None
        gold_score_matrix = sample = logprob = None
        judge_bank = object()

    with pytest.raises(BackendProtocolError, match="view_parameters"):
        check_model(Wrong())


def test_world_missing_gold_is_reported():
    class NoGold(ToyWorld):
        gold = None

    with pytest.raises(BackendProtocolError, match="gold"):
        check_world(NoGold())


def test_world_missing_an_attribute_is_reported():
    """`vocab` has a class-level default, so `del` on an instance cannot remove
    it; an object that simply never defines it is the honest fixture."""
    class Bare:
        prompt_len = 4

        def make_prompts(self, *a, **k): ...
        def correct_token(self, *a, **k): ...
        def gold(self, *a, **k): ...

    with pytest.raises(BackendProtocolError, match="vocab"):
        check_world(Bare())


def test_runner_rejects_an_incomplete_model_at_construction():
    """The whole point: fail before warm-up, not inside the first step."""
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    plan = make_method(cfg).plan()

    class Stub:
        def set_stage(self, s):
            pass

    with pytest.raises(BackendProtocolError):
        Runner(cfg, plan, SupervisionLedger(), model=Stub())


def test_runner_accepts_an_injected_model():
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    cfg.train.num_threads = 2
    plan = make_method(cfg).plan()
    model = _toy_model()
    runner = Runner(cfg, plan, SupervisionLedger(), model=model)
    assert runner.model is model
    # and the runner actually drives it
    ctx = runner.make_context(1, n_prompts=4, seed=1, audit_stage=1)
    out = runner.train_step(ctx, 1)
    assert "loss" in out and "adv" in out


def test_injected_model_and_world_can_differ_from_the_defaults():
    """A real backend supplies its own world; the runner must use it rather
    than silently falling back to the toy one."""
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    class TaggedWorld(ToyWorld):
        def make_prompts(self, stage, n, *, seed, offset=0):
            ids, tids = super().make_prompts(stage, n, seed=seed, offset=offset)
            return ids, [f"tagged:{t}" for t in tids]

    cfg = Config()
    plan = make_method(cfg).plan()
    world = TaggedWorld()
    runner = Runner(cfg, plan, SupervisionLedger(), world=world,
                    model=ToyModel(world, plan.spec, seed=0))
    assert runner.world is world
    _, tids = runner.stage_prompts(1, 2, seed=3)
    assert all(t.startswith("tagged:") for t in tids)


# ---------------------------------------------------------------------------
# L1 / L3 separation of the world's answer paths
# ---------------------------------------------------------------------------
#
# `correct_token` is the verifier's view; `demo_answer` is the demonstration's.
# In the toy world both are computed from the same frozen map, so the split
# changes no number -- but in a real backend they are different objects, and
# collapsing them lets an L1-labelled warm-up call the verifier while the
# ledger still reports zero gold reads (the call never passes through it).


class VerifierLockedWorld(ToyWorld):
    """A world whose verifier raises: only L1 paths may run against it."""

    def correct_token(self, stage, prompt_ids):
        raise AssertionError(
            "correct_token is the L3 verifier and must be unreachable from "
            "warm-up or demonstration code"
        )

    def gold(self, stage, prompt_ids, answers):
        raise AssertionError("gold must be unreachable from an L1 path")


def test_warmup_judge_never_touches_the_verifier():
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    plan = make_method(cfg).plan()
    world = VerifierLockedWorld()
    runner = Runner(cfg, plan, SupervisionLedger(), world=world,
                    model=ToyModel(world, plan.spec, seed=0))

    losses = runner.warmup_judge(1, n_rounds=3, n_per_round=4)
    assert len(losses) == 3
    assert all(isinstance(v, float) for v in losses)


def test_demonstration_batch_never_touches_the_verifier():
    from csr_grpo.methods.baselines import demonstration_batch

    world = VerifierLockedWorld()
    ids, targets = demonstration_batch(world, 1, 4, seed=0)
    assert ids.shape[0] == targets.shape[0] == 4


def test_supervised_step_never_touches_the_verifier():
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    cfg.method = "seq_sft"
    plan = make_method(cfg).plan()
    world = VerifierLockedWorld()
    runner = Runner(cfg, plan, SupervisionLedger(), world=world,
                    model=ToyModel(world, plan.spec, seed=0))
    ctx = runner.make_context(1, n_prompts=4, seed=1)
    out = runner.supervised_step(ctx, 1)
    assert out["supervised"] is True


def test_toy_world_still_agrees_on_both_paths():
    """The split must not change the toy numbers: both paths share the frozen
    map, so a difference here would be a silent behaviour change."""
    world = ToyWorld()
    ids, _ = world.make_prompts(2, 5, seed=3)
    assert torch.equal(world.correct_token(2, ids), world.demo_answer(2, ids))


def test_protocol_requires_the_demonstration_path():
    """Without `demo_answer` in the contract, a backend could satisfy every
    check while offering no verifier-free way to build demonstrations."""
    from csr_grpo.backends.protocol import REQUIRED_WORLD_METHODS

    assert "demo_answer" in REQUIRED_WORLD_METHODS

    class NoDemo(ToyWorld):
        demo_answer = None

    with pytest.raises(BackendProtocolError, match="demo_answer"):
        check_world(NoDemo())


# ---------------------------------------------------------------------------
# call-shape compatibility
# ---------------------------------------------------------------------------
#
# Presence of a method is not enough: HFBackend.sample originally took
# ``Sequence[str]`` and no ``generator``, so it satisfied every name check and
# still raised TypeError on the first training step (runner.py:242 calls
# ``sample(pid, G, generator=rng)`` with a token tensor).


def _complete_model(**overrides):
    """An object that satisfies every *name* check, so only signatures can fail."""
    class Complete:
        def view_parameters(self, view): ...
        def named_parameters(self): ...
        def parameters(self): ...
        def set_stage(self, stage): ...
        def actor_logits(self, prompt_ids, **kw): ...
        def judge_score_matrix(self, prompt_ids, answers, context_ids=None): ...
        def judge_prob(self, prompt_ids, answer_ids): ...
        def gold_score_matrix(self, prompt_ids, answers): ...
        def sample(self, prompt_ids, group_size, **kw): ...
        def logprob(self, sequences, attention_mask, prompt_len): ...
        judge_bank = object()

    for name, fn in overrides.items():
        setattr(Complete, name, fn)
    return Complete()


def test_protocol_checks_the_loop_call_shape_not_just_names():
    """The old HF signature: text in, no generator. Every name matches."""
    def text_sample(self, prompts, group_size, temperature=1.0):
        ...

    with pytest.raises(BackendProtocolError) as exc:
        check_model(_complete_model(sample=text_sample))

    msg = str(exc.value)
    assert "sample" in msg
    assert "generator" in msg


def test_hf_sample_accepts_the_loop_call_shape():
    """Drive the real HFBackend with a fake PEFT model -- no weights needed."""
    from csr_grpo.backends.hf import HFBackend

    class _Tok:
        pad_token_id = 0

    class _Model:
        def set_adapter(self, names):
            pass

        def generate(self, input_ids, attention_mask, **kw):
            n, t = input_ids.shape
            g = kw["num_return_sequences"]
            out = torch.cat([input_ids, torch.full((n, 2), 3, dtype=torch.long)], 1)
            return out.repeat_interleave(g, dim=0)

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            b, t = input_ids.shape
            return type("Out", (), {"logits": torch.zeros(b, t, 64)})()

    backend = HFBackend(model_name="x", spec=AdapterSpec.dual_adapter(r=8),
                        device="cpu")
    backend._model = _Model()
    backend._tokenizer = _Tok()
    # the exact call runner.py:242 makes
    seqs, mask, lp = backend.sample(torch.zeros(2, 4, dtype=torch.long), 3,
                                    generator=torch.Generator().manual_seed(0))
    assert seqs.shape[0] == 6
    assert mask.shape == seqs.shape


def test_hf_sample_leaves_the_global_rng_untouched():
    """Sampling must not perturb the global torch stream: other arms and the
    measurement code draw from it, and silently advancing it would break
    paired comparisons."""
    from csr_grpo.backends.hf import HFBackend

    class _Tok:
        pad_token_id = 0

    class _Model:
        def set_adapter(self, names):
            pass

        def generate(self, input_ids, attention_mask, **kw):
            n, t = input_ids.shape
            return torch.cat([input_ids, torch.full((n, 1), 3, dtype=torch.long)], 1)

        def __call__(self, input_ids, attention_mask):
            b, t = input_ids.shape
            return type("Out", (), {"logits": torch.zeros(b, t, 64)})()

    backend = HFBackend(model_name="x", spec=AdapterSpec.dual_adapter(r=8),
                        device="cpu")
    backend._model = _Model()
    backend._tokenizer = _Tok()

    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    backend.sample(torch.zeros(1, 4, dtype=torch.long), 2,
                   generator=torch.Generator().manual_seed(7))
    assert torch.equal(torch.get_rng_state(), before)


def test_call_shape_check_is_not_fooled_by_bound_methods():
    """`getattr(instance, name)` is already bound; binding it with an extra
    `self` would reject every correctly-shaped method.  ToyModel is the
    regression case: it was wrongly rejected while this check was written."""
    check_model(_toy_model())


def test_lone_star_and_kwargs_signatures_are_accepted():
    check_model(_complete_model())


# ---------------------------------------------------------------------------
# cfg.model.backend dispatch
# ---------------------------------------------------------------------------


def test_runner_defaults_to_toy_when_backend_is_toy():
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner
    from csr_grpo.backends.toy import ToyModel

    cfg = Config()
    cfg.model.backend = "toy"
    runner = Runner(cfg, make_method(cfg).plan(), SupervisionLedger())
    assert isinstance(runner.model, ToyModel)


def test_runner_requires_explicit_model_for_non_toy_backend():
    """When backend != toy, Runner cannot construct the backend itself: it
    needs a model path, tokeniser path, device assignments, and a gold verifier
    -- all caller-owned.  The error must be informative, not an AttributeError."""
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    cfg.model.backend = "hf"
    with pytest.raises(ValueError, match="model="):
        Runner(cfg, make_method(cfg).plan(), SupervisionLedger())


def test_runner_accepts_explicit_model_regardless_of_backend_field():
    """With ``model=`` supplied, cfg.model.backend is documentation only."""
    from csr_grpo.config.schema import Config
    from csr_grpo.methods.base import make_method
    from csr_grpo.supervision.ledger import SupervisionLedger
    from csr_grpo.training.runner import Runner

    cfg = Config()
    cfg.model.backend = "hf"      # would normally require injection
    plan = make_method(cfg).plan()
    model = _toy_model()           # a complete, check_model-passing object
    runner = Runner(cfg, plan, SupervisionLedger(), model=model)
    assert runner.model is model
