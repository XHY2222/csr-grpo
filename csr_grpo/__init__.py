"""Continual Self-Rewarding GRPO — controlled testbed.

Modules
-------
config       dataclass schema + YAML loader with strict validation
supervision  L1/L2/L3 ledger (runtime gold ban inside training scopes)
rewards      advantage construction (GRPO / rank-only / degenerate-zeroing)
diagnostics  fidelity metrics, 2x2 snapshot factorial, cluster bootstrap,
             matched-FPR early warning, dual measurement line
training     training-state snapshots (params + Adam + RNG) and the GRPO step
methods      Dual-Adapter, Replay, RLER-faithful, masking controls
experiments  E1 (switch vs continue + gold DID), E2 (protocol A/B intervention)
backends     toy CPU model and HF/LoRA model (lazy import)
"""

__version__ = "0.1.0"

from .supervision.ledger import (  # noqa: F401
    Level,
    Role,
    SupervisionLedger,
    SupervisionViolation,
)
