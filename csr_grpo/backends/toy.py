"""CPU toy backend: a real optimisation problem, small enough to test exactly.

There is no GPU on this machine, so every experiment must be runnable and
*meaningful* on CPU.  The toy world keeps the parts of the real setup that the
diagnostics depend on:

* **four heterogeneous stages**, each with its own frozen scoring map
  ``W_s ∈ R^{V×d}``; the gold answer for a prompt is ``argmax_y W_s[y]·h`` where
  ``h`` is the mean prompt embedding.  Task switching therefore genuinely moves
  the gold function, so forgetting and judge drift are emergent, not scripted;
* an **actor view** and a **judge view** that share a base network (frozen) and
  differ only through LoRA-style adapters — this is what makes ``r_S > 0``
  contamination possible and ``r_S = 0`` contamination-free;
* rollouts with per-token log probabilities, so the GRPO ratio and the frozen
  rollout buffer used by protocol A are the real objects.

Token layout: sequences are ``[p_0 … p_{L-1}, y]`` with ``L = 4`` prompt
tokens and a single completion token, ``attention_mask`` set to 1 only on the
completion (that is the loss mask).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..methods.lora import AdapterBank, AdapterSpec, build_bank, build_shared
from ..supervision.ledger import Level, Role, SupervisionLedger

VOCAB = 16
D_MODEL = 24
PROMPT_LEN = 4
STAGE_NAMES = ("logic", "code", "math", "instruction")


# ---------------------------------------------------------------------------
# world
# ---------------------------------------------------------------------------


@dataclass
class ToyWorld:
    """Frozen task structure: stage scoring maps and prompt generator."""

    vocab: int = VOCAB
    d_model: int = D_MODEL
    prompt_len: int = PROMPT_LEN
    n_stages: int = 4
    seed: int = 1234
    scale: float = 4.0
    _W: torch.Tensor | None = field(default=None, repr=False)
    _embed: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        g = torch.Generator().manual_seed(self.seed)
        self._W = torch.randn(self.n_stages, self.vocab, self.d_model,
                              generator=g) / (self.d_model ** 0.5)
        self._embed = torch.randn(self.vocab, self.d_model, generator=g) / 2.0

    # -- tensors ------------------------------------------------------------
    @property
    def W(self) -> torch.Tensor:
        assert self._W is not None
        return self._W

    @property
    def embed(self) -> torch.Tensor:
        assert self._embed is not None
        return self._embed

    def stage_index(self, stage: str | int) -> int:
        if isinstance(stage, int):
            return stage - 1 if stage >= 1 else stage
        return STAGE_NAMES.index(stage)

    # -- prompts ------------------------------------------------------------
    def make_prompts(self, stage: str | int, n: int, *, seed: int,
                     offset: int = 0) -> tuple[torch.Tensor, list[str]]:
        """Return ``(prompt_ids (n, L), task_ids)``."""
        s = self.stage_index(stage)
        g = torch.Generator().manual_seed(seed * 7919 + s * 104729 + offset)
        ids = torch.randint(0, self.vocab, (n, self.prompt_len), generator=g)
        task_ids = [f"s{s+1}-{seed}-{offset+i}" for i in range(n)]
        return ids, task_ids

    # -- gold ---------------------------------------------------------------
    def hidden(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        return self.embed[prompt_ids].mean(dim=1)

    def _answer_key(self, stage: str | int, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Private implementation shared by both answer paths (see below)."""
        s = self.stage_index(stage)
        h = self.hidden(prompt_ids)
        return torch.argmax(h @ self.W[s].t(), dim=-1)

    def correct_token(self, stage: str | int, prompt_ids: torch.Tensor) -> torch.Tensor:
        """The stage's gold answer — the verifier's view.

        This is the method a *real* backend implements by calling its L3
        verifier.  Nothing inside a training scope may call it; the ledger
        cannot see a verifier that was never routed through it.
        """
        return self._answer_key(stage, prompt_ids)

    def demo_answer(self, stage: str | int, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Answer attached to a demonstration — the L1 view.

        Separate from ``correct_token`` on purpose.  In the toy world both are
        computed from ``W_s``, so the split changes no number here; in a real
        backend they are genuinely different objects (a demo file vs. a solver),
        and collapsing them is how an L1-labelled warm-up ends up calling the
        verifier while the ledger still reports zero gold reads.
        """
        return self._answer_key(stage, prompt_ids)

    def gold(self, stage: str | int, prompt_ids: torch.Tensor,
             answers: torch.Tensor) -> torch.Tensor:
        """Gold correctness; ``answers`` is ``(P, G)`` or ``(N,)``.

        The correct token is per prompt, so the comparison broadcasts over the
        group axis.
        """
        correct = self._answer_key(stage, prompt_ids)      # (P,)
        a = answers.reshape(correct.shape[0], -1) if answers.dim() > 1 else answers
        if a.dim() == 1:
            return (a == correct).float()
        return (a == correct[:, None]).float()

    # -- judge features -----------------------------------------------------
    def judge_features(self, prompt_ids: torch.Tensor,
                       answer_ids: torch.Tensor) -> torch.Tensor:
        """Prompt context plus a weak binding of the candidate answer."""
        h = self.hidden(prompt_ids)
        e = self.embed[answer_ids]
        return torch.tanh(h + 0.7 * e)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class ToyModel(nn.Module):
    """Frozen base + one shared adapter + per-role private adapters."""

    def __init__(self, world: ToyWorld, spec: AdapterSpec,
                 seed: int = 0, judge_scale: float = 2.0,
                 mix: float = 0.85, context_scale: float = 0.5):
        super().__init__()
        self.world = world
        self.spec = spec
        self.mix = mix
        self.context_scale = context_scale
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("embed", world.embed.clone())
        self.register_buffer(
            "random_head",
            torch.randn(world.vocab, world.d_model, generator=g)
            / (world.d_model ** 0.5),
        )
        # fixed projection used only when a demonstration context is supplied
        self.register_buffer(
            "context_proj",
            torch.randn(world.d_model, world.d_model, generator=g)
            / (world.d_model ** 0.5),
        )
        self.register_buffer("judge_head",
                             torch.randn(world.vocab, world.d_model, generator=g)
                             / (world.d_model ** 0.5))
        # ONE shared adapter object, used by both views when r_S > 0.
        self.shared = build_shared(spec, world.d_model, world.vocab, generator=g)
        self.actor_bank = build_bank(spec, world.d_model, world.vocab, generator=g)
        self.judge_bank = build_bank(spec, world.d_model, world.vocab, generator=g)
        self.judge_scale = judge_scale
        self._configure_trainability()

    # -- adapters -----------------------------------------------------------
    def _configure_trainability(self) -> None:
        # All adapter parameters stay requires_grad=True: "frozen in stage" is
        # enforced by *which optimizer is stepped* (and by whether the training
        # scope touches the judge), not by breaking autograd.  That keeps
        # warm-up and the in-stage freeze compatible and, more importantly,
        # keeps the judge's gradient available for the E2 diagnostic that
        # measures what an update *would* have done.
        for mod in self._adapter_modules():
            if mod.A is None:
                continue
            mod.A.requires_grad_(True)
            mod.B.requires_grad_(True)

    def _adapter_modules(self):
        return (self.shared, getattr(self.actor_bank, "actor"),
                getattr(self.judge_bank, "judge"))

    def view_parameters(self, view: str) -> list[tuple[str, nn.Parameter]]:
        """Named trainable parameters of one view; ``shared`` appears in both."""
        if view not in {"actor", "judge"}:
            raise ValueError(view)
        active = self.spec.actor_active if view == "actor" else self.spec.judge_active
        out: list[tuple[str, nn.Parameter]] = []
        for name in active:
            if name == "shared":
                if self.shared.A is not None:
                    out += [("shared.A", self.shared.A), ("shared.B", self.shared.B)]
            elif name == "actor":
                mod = getattr(self.actor_bank, "actor")
                if mod.A is not None:
                    out += [("actor.A", mod.A), ("actor.B", mod.B)]
            elif name == "judge":
                mod = getattr(self.judge_bank, "judge")
                if mod.A is not None:
                    out += [("judge.A", mod.A), ("judge.B", mod.B)]
        return out


    def _delta(self, x: torch.Tensor, active) -> torch.Tensor:
        out = torch.zeros(*x.shape[:-1], self.world.vocab,
                          dtype=x.dtype, device=x.device)
        for name in active:
            if name == "shared":
                out = out + self.shared.delta(x)
            elif name == "actor":
                out = out + self.actor_bank.delta(x, ["actor"])
            elif name == "judge":
                out = out + self.judge_bank.delta(x, ["judge"])
            else:  # pragma: no cover
                raise ValueError(name)
        return out

    def actor_logits(self, prompt_ids: torch.Tensor,
                     context_ids: torch.Tensor | None = None,
                     stage_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Stage-conditional but *learnable* logits.

        The base mixes the stage's true map with a frozen random head::

            base = (1 - s) * scale * h @ W_sᵀ  +  s * h @ W_randᵀ

        Scaling ``h @ W_sᵀ`` alone cannot change the argmax (positive scaling
        preserves ranking), so a pure-scaled oracle stays an oracle. Mixing in a
        random head makes the pre-adapter policy a *weak but correlated* starting
        point (``s = 0.85`` gives roughly 35-50% accuracy), so the adapter has a
        real learning problem and different methods can diverge.

        ``context_ids`` gives the answer unconditioned on a demonstration
        context: the context's mean hidden state shifts the query hidden state
        through a fixed random projection.  Without it the toy actor is strictly
        per-row, so an in-context prefix (Row 2b / Row 4) could not influence
        anything and self-distillation would trivially equal the student
        (measured KL exactly 0).  ``None`` reproduces the previous behaviour
        bit-for-bit, so existing results are unchanged.
        """
        s_idx = getattr(self, "_active_stage", None)
        if s_idx is None:
            raise RuntimeError("set .set_stage(stage) before scoring")
        h = self.world.hidden(prompt_ids)
        if context_ids is not None and len(context_ids):
            # one context vector per query row (broadcast when a single block
            # is shared); scaled small so it conditions rather than overrides
            ctx = self.world.hidden(context_ids).mean(dim=0, keepdim=True)
            shift = torch.tanh(ctx @ self.context_proj) * self.context_scale
            if shift.shape[0] == 1 and h.shape[0] != 1:
                shift = shift.expand(h.shape[0], -1)
            h = h + shift
        if stage_ids is not None:
            # per-row stage: this is what static multi-task mixing needs, since
            # one step then contains prompts from several stages at once
            W = self.world.W[stage_ids]                       # (N, V, d)
            oracle = torch.einsum("nd,nvd->nv", h, W) * self.world.scale
        else:
            oracle = h @ self.world.W[s_idx].t() * self.world.scale
        random = h @ self.random_head.t() * self.world.scale
        base = (1.0 - self.mix) * oracle + self.mix * random
        return base + self._delta(h, self.spec.actor_active)

    def judge_logits(self, prompt_ids: torch.Tensor,
                     answer_ids: torch.Tensor,
                     context_ids: torch.Tensor | None = None) -> torch.Tensor:
        f = self.world.judge_features(prompt_ids, answer_ids)
        if context_ids is not None and len(context_ids):
            # Row 4 keeps theta_J frozen and conditions it in-context instead
            ctx = self.world.hidden(context_ids).mean(dim=0, keepdim=True)
            shift = torch.tanh(ctx @ self.context_proj) * self.context_scale
            if shift.shape[0] == 1 and f.shape[0] != 1:
                shift = shift.expand(f.shape[0], -1)
            f = f + shift
        base = (f @ self.judge_head.t()) * self.judge_scale
        return base + self._delta(f, self.spec.judge_active)


    def judge_prob(self, prompt_ids: torch.Tensor,
                   answer_ids: torch.Tensor) -> torch.Tensor:
        """Probability the answer token is the judge's preferred one, ``(N,)``.

        Backend-neutral companion to ``judge_logits``: the toy scores a whole
        answer vocabulary, a real GenRM judge scores ``[No, Yes]``.  Both
        return a ``(0, 1)`` score where higher means "better answer", so the
        weak-label BCE and the score matrix do not need per-backend branches.
        """
        logits = self.judge_logits(prompt_ids, answer_ids)
        return F.softmax(logits, dim=-1).gather(1, answer_ids.reshape(-1, 1)).squeeze(1)

    # -- stage context ------------------------------------------------------
    def set_stage(self, stage: str | int) -> None:
        self._active_stage = self.world.stage_index(stage)

    # -- scores -------------------------------------------------------------
    @torch.no_grad()
    def judge_score_matrix(self, prompt_ids: torch.Tensor,
                           answers: torch.Tensor,
                           context_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``(P, G)`` judge probability of each answer token."""
        p, g = answers.shape
        pid = prompt_ids.unsqueeze(1).expand(p, g, -1).reshape(p * g, -1)
        ans = answers.reshape(p * g)
        logits = self.judge_logits(pid, ans, context_ids=context_ids)
        probs = F.softmax(logits, dim=-1)
        return probs.gather(1, ans[:, None]).reshape(p, g)

    def gold_score_matrix(self, prompt_ids: torch.Tensor,
                          answers: torch.Tensor) -> torch.Tensor:
        """Gold correctness; the caller must already hold an L3 grant."""
        return self.world.gold(self._active_stage, prompt_ids, answers)

    # -- policy -------------------------------------------------------------
    @torch.no_grad()
    def sample(self, prompt_ids: torch.Tensor, group_size: int,
               temperature: float = 1.0,
               generator: torch.Generator | None = None,
               context_ids: torch.Tensor | None = None,
               stage_ids: torch.Tensor | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_logits(prompt_ids, context_ids, stage_ids) * temperature
        probs = F.softmax(logits, dim=-1)
        n = prompt_ids.shape[0]
        draws = torch.multinomial(probs.repeat_interleave(group_size, dim=0),
                                  1, generator=generator)
        answers = draws.squeeze(-1).view(n, group_size)
        seqs, mask = self._pack(prompt_ids, answers)
        logp = self.logprob(seqs, mask, prompt_ids.shape[1])
        return seqs, mask, logp

    def _pack(self, prompt_ids: torch.Tensor, answers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, g = answers.shape
        p = prompt_ids.unsqueeze(1).expand(n, g, -1).reshape(n * g, -1)
        a = answers.reshape(n * g, 1)
        seqs = torch.cat([p, a], dim=1)
        mask = torch.zeros_like(seqs)
        mask[:, -1] = 1.0
        return seqs, mask

    def logprob(self, sequences: torch.Tensor, attention_mask: torch.Tensor,
                prompt_len: int, context_ids: torch.Tensor | None = None
                ) -> torch.Tensor:
        """Per-token log probs, shape ``(N, T-1)``; prompt positions are 0."""
        prompt_ids = sequences[:, :prompt_len]
        answers = sequences[:, prompt_len]
        logits = self.actor_logits(prompt_ids, context_ids)
        lp = F.log_softmax(logits, dim=-1)
        token_lp = lp.gather(1, answers[:, None]).squeeze(1)      # (N,)
        out = torch.zeros(sequences.shape[0], sequences.shape[1] - 1)
        out[:, -1] = token_lp
        return out


class ToyPolicy:
    """Policy protocol adapter over ``ToyModel``."""

    def __init__(self, model: ToyModel):
        self.model = model

    def sample(self, task_ids, group_size, temperature=1.0, generator=None,
               prompt_ids: torch.Tensor | None = None):
        if prompt_ids is None:
            raise ValueError("toy policy needs prompt_ids")
        return self.model.sample(prompt_ids, group_size, temperature, generator)

    def logprob(self, sequences, attention_mask, prompt_len):
        return self.model.logprob(sequences, attention_mask, prompt_len)

    def named_parameters(self):
        return self.model.named_parameters()

    def parameters(self):
        return self.model.parameters()


# ---------------------------------------------------------------------------
# judge + verifier
# ---------------------------------------------------------------------------


class ToyJudge:
    """GenRM-shaped judge: greedy logit → probability of the answer token.

    Mirrors the paper's ``r = σ(z_Yes − z_No)``: a single logit difference
    squashed into ``(0, 1)``.  The adapter is what drifts across stages.
    """

    def __init__(self, model: ToyModel, trainable: bool = True):
        self.model = model
        self.trainable = trainable

    @torch.no_grad()
    def __call__(self, task_id: str, answers, prompt_id: int | torch.Tensor = 0):
        raise RuntimeError("use ToyJudge.score_rows / .score_matrix")

    @torch.no_grad()
    def score_matrix(self, prompt_ids: torch.Tensor,
                     answers: torch.Tensor) -> torch.Tensor:
        """``(P, G)`` judge probability of each answer token."""
        p, g = answers.shape
        pid = prompt_ids.unsqueeze(1).expand(p, g, -1).reshape(p * g, -1)
        ans = answers.reshape(p * g)
        return self.model.judge_prob(pid, ans).reshape(p, g)

    def loss(self, prompt_ids: torch.Tensor, answers: torch.Tensor,
             labels: torch.Tensor) -> torch.Tensor:
        """Route-A weak-label BCE (L1 supervision)."""
        p, g = answers.shape
        pid = prompt_ids.unsqueeze(1).expand(p, g, -1).reshape(p * g, -1)
        ans = answers.reshape(p * g)
        probs = self.model.judge_prob(pid, ans)
        return F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6),
                                      labels.reshape(-1).float())


class ToyVerifier:
    """Ledger-gated gold verifier backed by the world's true scoring map."""

    name = "toy_gold"

    def __init__(self, world: ToyWorld, ledger: SupervisionLedger,
                 stage: str | int = 1, role: str = Role.AUDIT_REPORTING):
        self.world = world
        self.ledger = ledger
        self.stage = stage
        self.role = role

    def verify(self, task_id: str, answers, purpose: str = "") -> list[float]:
        self.ledger.access(Level.L3, self.role, purpose or f"toy_gold:{task_id}")
        return self._verify_impl(task_id, answers)

    def _verify_impl(self, task_id: str, answers) -> list[float]:
        raise NotImplementedError("use verify_ids for tensors")

    def verify_ids(self, prompt_ids: torch.Tensor, answers: torch.Tensor,
                   purpose: str = "", role: str | None = None) -> torch.Tensor:
        self.ledger.access(Level.L3, role or self.role, purpose or "toy_gold:batch")
        return self.world.gold(self.stage, prompt_ids, answers)
