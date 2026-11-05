"""YAML → ``Config`` loading with dotted overrides and run manifests.

Presets live in ``configs/*.yaml``.  A preset names the method and stages; the
CLI can override any leaf with ``--set train.lr=3e-5``.  Every loaded config is
written back next to the run as ``config.resolved.json`` so a run is always
reproducible from its manifest alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import Config

_SECTIONS = ("model", "train", "judge", "data", "exp", "runtime")
_TOP = (
    "method", "reward_source", "stages_to_run", "is_reference",
    "is_diagnostic_arm", "confirmatory", "seed",
)


def _coerce_section(section: str, values: Mapping[str, Any]) -> Any:
    from . import schema as S

    cls = {
        "model": S.ModelConfig,
        "train": S.TrainingConfig,
        "judge": S.JudgeConfig,
        "data": S.DataConfig,
        "exp": S.ExperimentConfig,
        "runtime": S.RuntimeConfig,
    }[section]
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")
    kw = {}
    for k, v in values.items():
        if isinstance(v, list):
            v = tuple(v)
        kw[k] = v
    return cls(**kw)


def from_dict(raw: Mapping[str, Any]) -> Config:
    if not isinstance(raw, Mapping):
        raise TypeError("config must be a mapping")
    cfg = Config()
    for key, value in raw.items():
        if key in _SECTIONS:
            setattr(cfg, key, _coerce_section(key, value))
        elif key in _TOP:
            if key == "stages_to_run" and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
        else:
            raise ValueError(f"unknown top-level config key {key!r}")
    return cfg


# YAML 1.1 -- the parser PyYAML implements -- requires a dot in the mantissa
# before an exponent, so ``3e-5`` loads as the *string* ``'3e-5'``.  Command
# line overrides are written the way people write floats (``--set
# train.lr=3e-5``), and the string then reached the optimizer, which raised
# ``TypeError: '<=' not supported between 'float' and 'str'`` only after the
# model had been built.  Match the plain-language numeric forms ourselves.
_NUMERIC = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_scalar(text: str) -> Any:
    """Parse one override value, preferring numbers over YAML's quirks."""
    stripped = text.strip()
    if _NUMERIC.match(stripped):
        as_float = float(stripped)
        # keep integers as integers: ``--set seed=7`` must not become 7.0
        if as_float.is_integer() and "." not in stripped and "e" not in stripped.lower():
            return int(as_float)
        return as_float
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: Config, overrides: list[str] | tuple[str, ...]) -> Config:
    """Apply ``section.key=value`` overrides in place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like a.b=value")
        path, value = item.split("=", 1)
        parts = path.split(".")
        if len(parts) == 1:
            if parts[0] not in _TOP:
                raise ValueError(f"unknown top-level key {parts[0]!r}")
            setattr(cfg, parts[0], _parse_scalar(value))
            continue
        section, key = parts[0], parts[1]
        if section not in _SECTIONS:
            raise ValueError(f"unknown section {section!r}")
        obj = getattr(cfg, section)
        if not any(f.name == key for f in fields(obj)):
            raise ValueError(f"unknown key {section}.{key}")
        setattr(obj, key, _parse_scalar(value))
    return cfg


def load(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = from_dict(raw)
    apply_overrides(cfg, overrides)
    return cfg.validate()


def save_manifest(cfg: Config, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.resolved.json"
    path.write_text(json.dumps(cfg.to_dict(), indent=2, default=list), encoding="utf-8")
    return path


def describe(cfg: Config) -> str:
    """Short human-readable line used in logs."""
    gold = "reference" if cfg.is_reference else (
        "diagnostic" if cfg.is_diagnostic_arm else "trained"
    )
    return (
        f"method={cfg.method} reward={cfg.reward_source} gold={gold} "
        f"stages={cfg.stages_to_run} seed={cfg.seed}"
    )
