"""HuggingFace / LoRA backend for cluster runs.

The dev machine has no GPU, so this module is written to be *importable and
partially testable* on CPU while its heavy paths run on Cybertron.  The parts
that are pure logic -- mapping ``AdapterSpec`` onto named PEFT adapters, the
completion mask, and the GenRM logit-difference reward -- are unit-tested here;
the parts that need weights are exercised on the cluster.

Design
------
The paper needs three adapters with different sharing rules:

================  ==========================================
``shared``        attached to *both* views when ``r_S > 0``
``actor``         private to the actor view
``judge``         private to the judge view
================  ==========================================

PEFT supports several named adapters on one model and can activate a subset,
so the two views are ``set_adapter(["shared", "actor"])`` and
``set_adapter(["shared", "judge"])``.  Under ``r_S = 0`` the shared adapter is
simply never created, which gives Dual-Adapter its disjointness property.

Freezing is by *optimizer construction*, exactly as in the toy backend: the
judge adapters stay trainable tensor-wise so that warm-up can train them and so
that E2 can measure what a judge update would do; "frozen inside a stage" means
the trainer never steps the judge optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from ..methods.lora import AdapterSpec
from . import register_backend

# Heavy dependencies are imported lazily so `import csr_grpo` works anywhere.
try:  # pragma: no cover - import availability only
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    _HF_AVAILABLE = True
except Exception:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore
    LoraConfig = TaskType = get_peft_model = None  # type: ignore
    _HF_AVAILABLE = False


def hf_available() -> bool:
    return _HF_AVAILABLE


# ---------------------------------------------------------------------------
# pure logic (unit-tested on CPU)
# ---------------------------------------------------------------------------


def lora_configs(spec: AdapterSpec,
                 target_modules: Sequence[str]) -> dict[str, dict]:
    """Map the paper's ``AdapterSpec`` onto per-adapter PEFT kwargs.

    ``r_S = 0`` yields **no** ``shared`` entry at all rather than a zero-rank
    adapter: PEFT has no meaningful rank-0 LoRA, and "no parameters" is exactly
    the semantics the contamination argument needs.
    """
    out: dict[str, dict] = {}
    if spec.r_shared > 0:
        out["shared"] = {
            "r": int(spec.r_shared),
            "lora_alpha": int(spec.alpha),
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": list(target_modules),
            "task_type": "CAUSAL_LM",
        }
    if spec.r_actor_private > 0:
        out["actor"] = {
            "r": int(spec.r_actor_private),
            "lora_alpha": int(spec.alpha),
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": list(target_modules),
            "task_type": "CAUSAL_LM",
        }
    if spec.r_judge_private > 0:
        out["judge"] = {
            "r": int(spec.r_judge_private),
            "lora_alpha": int(spec.alpha),
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": list(target_modules),
            "task_type": "CAUSAL_LM",
        }
    return out


def activation_for(spec: AdapterSpec, view: str) -> list[str]:
    """Which named adapters are active for a view, filtered to those created."""
    if view not in {"actor", "judge"}:
        raise ValueError(view)
    wanted = spec.actor_active if view == "actor" else spec.judge_active
    available = set(lora_configs(spec, ("x",)))
    return [name for name in wanted if name in available]


def completion_mask(sequences: torch.Tensor, prompt_len: int,
                    pad_token_id: int | None = None) -> torch.Tensor:
    """Loss mask: 1 on completion tokens, 0 on the prompt and on padding.

    Matches the toy backend's convention (``sequences`` is ``(N, T)`` and the
    prompt occupies ``[0, prompt_len)``) so ``grpo_loss`` can consume either.
    """
    mask = torch.zeros_like(sequences, dtype=torch.float32)
    mask[:, prompt_len:] = 1.0
    if pad_token_id is not None:
        mask = mask * (sequences != pad_token_id).float()
    return mask


def genrm_reward(yes_logit: torch.Tensor, no_logit: torch.Tensor) -> torch.Tensor:
    """``r = σ(z_Yes − z_No) ∈ (0, 1)`` — the paper's deterministic GenRM head.

    Computed on raw last-layer logits rather than parsed verdict text, so the
    reward is continuous and has no generation variance at temperature 0.
    """
    return torch.sigmoid(yes_logit.float() - no_logit.float())


def gather_completion_logprob(logits: torch.Tensor, sequences: torch.Tensor,
                              prompt_len: int) -> torch.Tensor:
    """Per-token log probs of the completion, shape ``(N, T − prompt_len)``.

    ``logits`` must already be shifted (``logits[:, :-1]``) so that position
    ``t`` predicts token ``t+1``.
    """
    comp = sequences[:, prompt_len:]
    if logits.shape[1] != comp.shape[1]:
        raise ValueError(
            f"logits time axis {logits.shape[1]} != completion length {comp.shape[1]}"
        )
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(2, comp.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# model holder
# ---------------------------------------------------------------------------

class _ParamView(torch.nn.Module):
    """Parameter container exposing a borrowed set of tensors.

    ``ModuleState.capture``/``restore`` need an ``nn.Module`` whose
    ``named_parameters()`` are the judge adapters.  Registering the *same*
    tensor objects (not copies) is what makes ``restore`` write back into the
    live model.
    """

    def __init__(self, params: dict[str, torch.Tensor]):
        super().__init__()
        for i, (name, p) in enumerate(params.items()):
            # dots are illegal in module attribute names; keep a stable order
            self.register_parameter(f"p{i}", torch.nn.Parameter(p, requires_grad=True))
        self._names = list(params)

    def named_parameters(self, *args, **kwargs):
        for i, name in enumerate(self._names):
            yield name, getattr(self, f"p{i}")




@dataclass
class HFBackend:
    """A base LM with three named LoRA adapters and two views."""

    model_name: str
    spec: AdapterSpec
    target_modules: Sequence[str] = ("q_proj", "v_proj")
    dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = False
    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)
    _view: str = "actor"
    _stage: str | int | None = field(default=None, repr=False)
    _sample_calls: int = field(default=0, repr=False)
    _gold_verifier: object | None = field(default=None, repr=False)
    _judge_bank_view: object | None = field(default=None, repr=False)

    # -- construction -------------------------------------------------------
    def load(self) -> "HFBackend":
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "transformers/peft are unavailable; HFBackend needs the cluster image"
            )
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[self.dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code).to(self.device)

        configs = lora_configs(self.spec, self.target_modules)
        # create the first adapter inline, attach the rest by name
        names = list(configs)
        if not names:
            raise ValueError("AdapterSpec produced no LoRA adapters")
        self._model = get_peft_model(base, LoraConfig(**configs[names[0]],
                                                      inference_mode=False),
                                     adapter_name=names[0])
        for name in names[1:]:
            self._model.add_adapter(name, LoraConfig(**configs[name],
                                                     inference_mode=False))
        self.set_view("actor")
        return self

    # -- views --------------------------------------------------------------
    def set_view(self, view: str) -> None:
        active = activation_for(self.spec, view)
        if not active:
            raise RuntimeError(f"view {view!r} has no active adapters")
        self._model.set_adapter(active)
        self._view = view

    def view_parameters(self, view: str) -> list[tuple[str, torch.nn.Parameter]]:
        """Named parameters of one view, mirroring the toy backend's API."""
        active = set(activation_for(self.spec, view))
        out: list[tuple[str, torch.nn.Parameter]] = []
        for name, p in self._model.named_parameters():
            if not p.requires_grad:
                continue
            # PEFT names adapters as ...lora_A.<adapter_name>.weight
            if any(f".{a}." in name or name.endswith(f".{a}") for a in active):
                out.append((name, p))
        return out

    # -- Policy protocol ----------------------------------------------------
    @torch.no_grad()
    def sample(self, prompt_ids: torch.Tensor, group_size: int,
               temperature: float = 1.0,
               generator: torch.Generator | None = None,
               context_ids: torch.Tensor | None = None,
               max_new_tokens: int = 256,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out ``group_size`` completions per prompt (actor view).

        Takes **token ids**, not text, because the training loop holds ids and
        needs the sampling RNG to advance in step with ``runner.rng``; taking
        strings here would make every experiment's rollouts irreproducible.
        ``temperature``/``context_ids`` are accepted for protocol parity with
        the toy backend (HF conditions on text, so a context block must be part
        of ``prompt_ids``).

        The returned mask is a *loss* mask -- 1 on completion tokens only --
        matching what ``grpo_loss`` expects from the toy backend.
        """
        if context_ids is not None and len(context_ids):
            raise NotImplementedError(
                "HF conditioning is textual: prepend demonstrations to the prompt"
            )
        self.set_view("actor")

        # transformers samples from the global torch RNG; derive a per-call
        # seed from runner.rng so paired arms see identical rollouts and the
        # global stream is left exactly as it was found.
        state = torch.get_rng_state()
        try:
            if generator is not None:
                seed = (int(generator.initial_seed()) + 7919 * self._sample_calls) % (2 ** 63 - 1)
                torch.manual_seed(seed)
            self._sample_calls += 1
            prompt_len = int(prompt_ids.shape[1])
            gen = self._model.generate(
                input_ids=prompt_ids.to(self.device),
                attention_mask=torch.ones_like(prompt_ids).to(self.device),
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                num_return_sequences=int(group_size),
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            loss_mask = completion_mask(gen, prompt_len,
                                        pad_token_id=self._tokenizer.pad_token_id)
            old_lp = self.logprob(gen, torch.ones_like(gen), prompt_len)
        finally:
            torch.set_rng_state(state)
        return gen, loss_mask, old_lp

    def logprob(self, sequences: torch.Tensor, attention_mask: torch.Tensor,
                prompt_len: int) -> torch.Tensor:
        """Completion log probs, shape ``(N, T − prompt_len)``."""
        self.set_view("actor")
        out = self._model(input_ids=sequences, attention_mask=attention_mask)
        # position t predicts token t+1, so drop the last logit and the first token
        logits = out.logits[:, prompt_len - 1: -1, :]
        return gather_completion_logprob(logits, sequences, prompt_len)

    def named_parameters(self):
        return self._model.named_parameters()

    def parameters(self):
        return self._model.parameters()

    # -- GenRM judge --------------------------------------------------------
    def judge_scores(self, rubric_prompts: Sequence[str],
                     answers: Sequence[str]) -> torch.Tensor:
        """``σ(z_Yes − z_No)`` for each (prompt, answer) pair.

        Real runs generate a short greedy rubric CoT (≤192 tokens) ending in a
        verdict, then read the last-layer logit difference at the verdict
        position.  This method takes pre-built rubric prompts so the prompt
        template stays in the experiment layer where it is auditable.
        """
        self.set_view("judge")
        yes_id = self._yes_no_ids("Yes")
        no_id = self._yes_no_ids("No")
        scores: list[torch.Tensor] = []
        for prompt, answer in zip(rubric_prompts, answers):
            text = prompt.format(answer=answer)
            enc = self._tokenizer(text, return_tensors="pt").to(self.device)
            out = self._model(**enc)
            last = out.logits[:, -1, :].float()
            scores.append(genrm_reward(last[:, yes_id], last[:, no_id]).squeeze(0))
        return torch.stack(scores)

    def _yes_no_ids(self, word: str) -> int:
        ids = self._tokenizer(word, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"tokenizer produced no ids for {word!r}")
        return int(ids[0])

    # ------------------------------------------------------------------
    # Runner protocol (the six methods training/runner.py calls)
    # ------------------------------------------------------------------
    # The toy backend scores single answer tokens; a real LM produces whole
    # completions.  These adapters keep the *training loop* unchanged by
    # making the backend do the text conversion, rather than pushing
    # tokenizer knowledge up into the runner.

    def set_stage(self, stage: str | int) -> None:
        """Record the active stage.

        An HF model has no stage conditioning of its own: the stage enters
        through the prompts and through which LoRA adapters are trained.  The
        attribute exists so the runner can call the same protocol on either
        backend, and so a subclass can use it (e.g. a stage-conditioned prefix).
        """
        self._stage = stage

    @torch.no_grad()
    def actor_logits(self, prompt_ids: torch.Tensor,
                     context_ids: torch.Tensor | None = None,
                     stage_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Next-token logits ``(N, V)`` for unbatched-left-padded prompts.

        ``context_ids``/``stage_ids`` are accepted for protocol compatibility;
        a real LM conditions on the text itself, so an explicit context block
        must be prepended to ``prompt_ids`` by the caller.
        """
        if context_ids is not None and len(context_ids):
            raise NotImplementedError(
                "HF conditioning is textual: prepend demonstrations to the "
                "prompt instead of passing context_ids"
            )
        self.set_view("actor")
        out = self._model(input_ids=prompt_ids)
        return out.logits[:, -1, :].float()

    def judge_logits(self, prompt_ids: torch.Tensor,
                     answer_ids: torch.Tensor,
                     context_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Two-class ``(N, 2)`` GenRM logits ``[z_No, z_Yes]``.

        The toy backend returns ``(N, V)`` logits over answer tokens; a real
        judge answers a yes/no question about a completion, so the output
        dimension is 2 rather than the vocabulary.  ``judge_prob`` bridges the
        difference for callers that only want a probability.
        """
        if context_ids is not None and len(context_ids):
            raise NotImplementedError("HF judge takes its context in the prompt text")
        self.set_view("judge")
        yes_id = self._yes_no_ids("Yes")
        no_id = self._yes_no_ids("No")
        texts = self._decode_pairs(prompt_ids, answer_ids)
        rows: list[torch.Tensor] = []
        for text in texts:
            enc = self._tokenizer(text, return_tensors="pt").to(self.device)
            last = self._model(**enc).logits[:, -1, :].float()
            rows.append(torch.stack([last[:, no_id], last[:, yes_id]], dim=-1))
        return torch.cat(rows, dim=0)

    def judge_prob(self, prompt_ids: torch.Tensor,
                   answer_ids: torch.Tensor) -> torch.Tensor:
        """``σ(z_Yes − z_No)`` per row — backend-neutral judge score.

        Toy: the softmax probability of the answer token.  HF: the GenRM head.
        Both land in ``(0, 1)`` with the same "higher is better" direction, so
        the weak-label BCE in ``Runner.warmup_judge`` is backend-independent.
        """
        logits = self.judge_logits(prompt_ids, answer_ids)
        return torch.softmax(logits, dim=-1)[:, 1]

    def judge_score_matrix(self, prompt_ids: torch.Tensor,
                           answers: torch.Tensor,
                           context_ids: torch.Tensor | None = None
                           ) -> torch.Tensor:
        """``(P, G)`` judge probability, matching the toy backend's shape."""
        p, g = answers.shape
        pid = prompt_ids.unsqueeze(1).expand(p, g, -1).reshape(p * g, -1)
        ans = answers.reshape(p * g)
        return self.judge_prob(pid, ans).reshape(p, g)

    def gold_score_matrix(self, prompt_ids: torch.Tensor,
                          answers: torch.Tensor) -> torch.Tensor:
        """Gold correctness ``(P, G)``; the caller must already hold an L3 grant.

        A real verifier is domain-specific, so it is injected rather than
        guessed.  Refusing loudly is deliberate: a silent zero matrix would be
        indistinguishable from "every answer is wrong" and would corrupt every
        metric downstream.
        """
        if self._gold_verifier is None:
            raise RuntimeError(
                "HFBackend has no gold verifier; pass gold_verifier=(prompts, "
                "answers) -> (P, G) tensor. A stub here would be a silent leak."
            )
        return self._gold_verifier(prompt_ids, answers)

    @property
    def judge_bank(self) -> torch.nn.Module:
        """The judge-side parameters as one module, for snapshot/restore.

        ``ModuleState`` walks ``named_parameters()``, so this view must expose
        exactly the judge adapters and nothing else.
        """
        if self._judge_bank_view is None:
            self._judge_bank_view = _ParamView(
                {n: p for n, p in self.view_parameters("judge")})
        return self._judge_bank_view

    def _decode_pairs(self, prompt_ids: torch.Tensor,
                      answer_ids: torch.Tensor) -> list[str]:
        """Render ``(prompt, answer)`` token rows as judge prompt text."""
        prompts = self._tokenizer.batch_decode(
            prompt_ids.detach().to("cpu"), skip_special_tokens=True)
        answers = self._tokenizer.batch_decode(
            answer_ids.reshape(-1, 1).detach().to("cpu"), skip_special_tokens=True)
        return [t + "\n" + a for t, a in zip(prompts, answers)]


@register_backend("hf")
def build_hf_backend(**kwargs) -> HFBackend:  # pragma: no cover - cluster only
    return HFBackend(**kwargs).load()
