"""HF backend *logic* tests (no weights, no GPU).

Only the parts whose correctness is independent of the model are asserted here:
adapter construction, view activation, completion masking, the GenRM head, and
log-prob gathering.  The weight-dependent paths run on the cluster.
"""

import numpy as np
import pytest
import torch

from csr_grpo.backends import available_backends, build_backend
from csr_grpo.backends.hf import (
    HFBackend, activation_for, completion_mask, gather_completion_logprob,
    genrm_reward, hf_available, lora_configs,
)
from csr_grpo.methods.lora import AdapterSpec


# ---------------------------------------------------------------------------
# adapter construction
# ---------------------------------------------------------------------------


def test_dual_adapter_creates_no_shared_adapter():
    """r_S = 0 must yield *no* shared entry, not a rank-0 one."""
    cfg = lora_configs(AdapterSpec.dual_adapter(r=64), ("q_proj", "v_proj"))
    assert "shared" not in cfg
    assert cfg["actor"]["r"] == 64 and cfg["judge"]["r"] == 64


def test_shared_architecture_creates_all_three():
    cfg = lora_configs(AdapterSpec.shared(r=32, r_private=32), ("q_proj",))
    assert set(cfg) == {"shared", "actor", "judge"}
    assert cfg["shared"]["r"] == 32


def test_masking_only_architecture_has_no_private_branch():
    spec = AdapterSpec.shared(r=32, r_private=0)
    cfg = lora_configs(spec, ("q_proj",))
    assert set(cfg) == {"shared"}


def test_activation_excludes_absent_adapters():
    dual = AdapterSpec.dual_adapter(r=8)
    assert activation_for(dual, "actor") == ["actor"]
    assert activation_for(dual, "judge") == ["judge"]
    shared = AdapterSpec.shared(r=8, r_private=8)
    assert activation_for(shared, "actor") == ["shared", "actor"]
    assert activation_for(shared, "judge") == ["shared", "judge"]


def test_activation_rejects_unknown_view():
    with pytest.raises(ValueError):
        activation_for(AdapterSpec.dual_adapter(r=8), "critic")


def test_shared_views_overlap_only_on_shared():
    """The contamination path is exactly the intersection of the two views."""
    spec = AdapterSpec.shared(r=8, r_private=8)
    a, j = set(activation_for(spec, "actor")), set(activation_for(spec, "judge"))
    assert a & j == {"shared"}
    dual = AdapterSpec.dual_adapter(r=8)
    assert set(activation_for(dual, "actor")) & set(activation_for(dual, "judge")) == set()


# ---------------------------------------------------------------------------
# masks and rewards
# ---------------------------------------------------------------------------


def test_completion_mask_covers_only_completion():
    seq = torch.tensor([[5, 6, 7, 8, 9]])
    m = completion_mask(seq, prompt_len=3)
    assert m.tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0]]


def test_completion_mask_drops_padding():
    seq = torch.tensor([[5, 6, 7, 0, 0]])
    m = completion_mask(seq, prompt_len=3, pad_token_id=0)
    assert m.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0]]


def test_genrm_reward_is_sigmoid_of_logit_difference():
    yes = torch.tensor([2.0, 0.0])
    no = torch.tensor([0.0, 0.0])
    r = genrm_reward(yes, no)
    # compare tensor-to-tensor: this host's torch/numpy bridge is broken
    assert torch.isclose(r[0], torch.tensor(1 / (1 + np.exp(-2.0))), atol=1e-6)
    assert torch.isclose(r[1], torch.tensor(0.5))
    assert torch.all((r > 0) & (r < 1))


def test_gather_completion_logprob_matches_manual():
    # vocab 4, completion length 2
    logits = torch.log(torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                                      [0.4, 0.3, 0.2, 0.1]]]))
    seq = torch.tensor([[9, 9, 2, 0]])          # completion = [2, 0]
    got = gather_completion_logprob(logits, seq, prompt_len=2)
    expected = torch.log(torch.tensor([[0.3, 0.4]]))
    assert torch.allclose(got, expected, atol=1e-6)


def test_gather_rejects_shape_mismatch():
    logits = torch.zeros(1, 3, 4)
    seq = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        gather_completion_logprob(logits, seq, prompt_len=1)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_lists_toy_always():
    assert "toy" in available_backends()


def test_build_backend_toy_returns_model():
    model = build_backend("toy", seed=0)
    assert hasattr(model, "actor_logits")


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        build_backend("nope")


def test_hf_dependencies_are_optional():
    """The package must import and test cleanly whether or not peft exists."""
    assert isinstance(hf_available(), bool)


# ---------------------------------------------------------------------------
# Runner protocol (added for cluster runs; testable without weights)
# ---------------------------------------------------------------------------
#
# The four methods below are what training/runner.py calls on any backend.
# `load()` needs real weights, so these tests inject a fake tokenizer and a
# fake PEFT model with the same *shape* of interface.  That pins the protocol
# itself -- shapes, directions, failure modes -- which is the part that can
# silently corrupt a cluster run.


YES_ID, NO_ID = 11, 12


class _Enc(dict):
    """Tokenizer output with a no-op ``.to()`` (everything is already CPU)."""

    def to(self, device):
        return self


class _Out:
    """Model output: attribute access, like a HF ``ModelOutput``."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        if not add_special_tokens and text in {"Yes", "No"}:
            return {"input_ids": [YES_ID if text == "Yes" else NO_ID]}
        ids = [(ord(c) % 7) + 20 for c in text][:8] or [20]
        return _Enc(input_ids=torch.tensor([ids]),
                    attention_mask=torch.ones(1, len(ids), dtype=torch.long))

    def batch_decode(self, rows, skip_special_tokens=True):
        if rows.dim() == 1:
            rows = rows.unsqueeze(-1)
        out = []
        for r in rows.tolist():
            out.append("".join(chr(97 + (i % 26)) for i in r))
        return out


class _FakeModel(torch.nn.Module):
    """Minimal PEFT-shaped model: named adapters + a controllable verdict."""

    def __init__(self, spec, vocab=64):
        super().__init__()
        self.vocab = vocab
        self.z_yes, self.z_no = 1.0, -1.0
        self.active: list[str] = []
        names = [n for n in ("shared", "actor", "judge")
                 if n in lora_configs(spec, ("x",))]
        self.adapter = torch.nn.ModuleDict({
            n: torch.nn.ParameterDict(
                {"A": torch.nn.Parameter(torch.full((2,), 0.1)),
                 "B": torch.nn.Parameter(torch.full((2,), 0.2))})
            for n in names})
        self.base = torch.nn.Linear(1, 1, bias=False)
        self.base.weight.requires_grad_(False)

    def set_adapter(self, names):
        self.active = list(names)

    def forward(self, input_ids=None, attention_mask=None):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab)
        logits[:, -1, YES_ID] = self.z_yes
        logits[:, -1, NO_ID] = self.z_no
        logits[:, -1, 0] = 0.5
        return _Out(logits=logits)


def _fake_backend(spec=None):
    spec = spec or AdapterSpec.dual_adapter(r=8, alpha=16.0)
    be = HFBackend(model_name="fake", spec=spec, device="cpu")
    be._model = _FakeModel(spec)
    be._tokenizer = _FakeTokenizer()
    be.set_view("actor")
    return be


def test_hf_judge_logits_are_two_class_not_vocabulary():
    """A real judge answers yes/no; the toy scores a token vocabulary.  The
    output width is the contract callers rely on."""
    be = _fake_backend()
    pid = torch.tensor([[20, 21], [22, 23]])
    ans = torch.tensor([[0], [1]])
    logits = be.judge_logits(pid, ans)
    assert logits.shape == (2, 2)
    # float(): pytest.approx on a 0-dim tensor calls Tensor.numpy(), which is
    # unavailable on this host (torch built against NumPy 1.x).
    assert float(logits[0, 0]) == pytest.approx(float(be._model.z_no))
    assert float(logits[0, 1]) == pytest.approx(float(be._model.z_yes))


def test_hf_judge_prob_is_the_genrm_sigmoid():
    be = _fake_backend()
    be._model.z_yes, be._model.z_no = 2.5, -0.5
    prob = be.judge_prob(torch.tensor([[20]]), torch.tensor([0]))
    expected = float(torch.sigmoid(torch.tensor(2.5 - (-0.5))))
    assert float(prob[0]) == pytest.approx(expected, rel=1e-6)


def test_hf_judge_prob_tracks_preference_direction():
    be = _fake_backend()
    pid, ans = torch.tensor([[20]]), torch.tensor([0])
    be._model.z_yes, be._model.z_no = -3.0, 3.0
    low = float(be.judge_prob(pid, ans)[0])
    be._model.z_yes, be._model.z_no = 3.0, -3.0
    high = float(be.judge_prob(pid, ans)[0])
    assert low < 0.5 < high


def test_hf_judge_score_matrix_shape_matches_toy_convention():
    be = _fake_backend()
    p, g = 3, 4
    pid = torch.arange(20, 20 + p * 3).reshape(p, 3)
    ans = (torch.arange(p * g).reshape(p, g)) % 3
    mat = be.judge_score_matrix(pid, ans)
    assert mat.shape == (p, g)
    assert float(mat.min()) >= 0.0 and float(mat.max()) <= 1.0


def test_hf_gold_scores_refuse_to_be_invented():
    """A stub zero matrix would read as 'every answer is wrong' and poison
    every downstream metric, so the backend must refuse instead."""
    be = _fake_backend()
    with pytest.raises(RuntimeError):
        be.gold_score_matrix(torch.tensor([[20]]), torch.tensor([[0]]))


def test_hf_gold_scores_use_the_injected_verifier():
    be = _fake_backend()
    be._gold_verifier = lambda pid, ans: torch.ones(ans.shape)
    mat = be.gold_score_matrix(torch.tensor([[20, 21]]), torch.tensor([[1, 0]]))
    assert mat.shape == (1, 2)
    assert torch.equal(mat, torch.ones(1, 2))


def test_hf_actor_logits_return_next_token_logits():
    be = _fake_backend()
    logits = be.actor_logits(torch.tensor([[20, 21, 22]]))
    assert logits.shape == (1, be._model.vocab)
    assert float(logits[0, 0]) == pytest.approx(0.5)


def test_hf_actor_logits_reject_context_ids():
    """HF conditioning is textual; silently ignoring context_ids would make
    Row 4/2b look implemented while conditioning nothing."""
    be = _fake_backend()
    with pytest.raises(NotImplementedError):
        be.actor_logits(torch.tensor([[20]]), context_ids=torch.tensor([[21]]))


def test_hf_judge_bank_round_trips_through_module_state():
    from csr_grpo.training.state import ModuleState

    be = _fake_backend()
    bank = be.judge_bank
    names = [n for n, _ in bank.named_parameters()]
    assert len(names) == 2 and all(".judge." in n for n in names)

    snapshot = ModuleState.capture(bank)
    original = {n: float(p[0]) for n, p in be.view_parameters("judge")}
    assert original["adapter.judge.A"] != original["adapter.judge.B"], \
        "the fixture must give A and B distinct values or a mixing bug hides"
    with torch.no_grad():
        for _, p in be.view_parameters("judge"):
            p.add_(1.0)
    snapshot.restore(bank)
    restored = {n: float(p[0]) for n, p in be.view_parameters("judge")}
    assert restored == pytest.approx(original)


def test_hf_view_parameters_exclude_the_frozen_base():
    be = _fake_backend()
    actor = dict(be.view_parameters("actor"))
    judge = dict(be.view_parameters("judge"))
    assert actor and judge
    assert not set(actor) & set(judge)      # r_S = 0 -> fully disjoint
    assert all("base" not in n for n in list(actor) + list(judge))
