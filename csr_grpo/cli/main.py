"""``python -m csr_grpo.cli <command>`` — humans and the cluster submit script.

Commands
--------
``plan``        print the method table and the supervision contract
``run``         train one method through the configured stages
``e1``          Switch vs. Continue + gold DID (optionally the E1' held-out pair)
``e2``          Protocol A / B reward-source intervention
``ledger``      re-print an audit log written by an earlier run
``selftest``    run the fast test suite (``pytest tests -x -q``)

Every command writes a run manifest (``config.resolved.json``), a ledger log
(``ledger.jsonl``) and a JSON result file, so a reported number can always be
traced back to the exact configuration and to the number of gold reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import loader
from ..config.schema import Config
from ..supervision.ledger import SupervisionLedger, SupervisionViolation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _prepare(cfg: Config, out_dir: Path) -> tuple[SupervisionLedger, Path]:
    import torch

    # The toy tensors are tiny (G=8).  Letting PyTorch use every core makes
    # OpenMP contention dominate the actual work — on a 192-core host an E2 run
    # that takes 1.4 s with 4 threads did not finish in 115 s without this.
    torch.set_num_threads(max(1, int(cfg.runtime.num_threads)))

    out_dir.mkdir(parents=True, exist_ok=True)
    loader.save_manifest(cfg, out_dir)
    ledger = SupervisionLedger(strict=cfg.runtime.ledger_strict)
    return ledger, out_dir


def _finish(ledger: SupervisionLedger, out_dir: Path, payload: dict,
            *, allow_gold_control: bool = False) -> int:
    ledger.dump(out_dir / "ledger.jsonl")
    summary = ledger.summary()
    payload = {**payload, "supervision": summary}
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2,
                                                    default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    try:
        ledger.assert_clean(allow_gold_control=allow_gold_control)
    except SupervisionViolation as exc:
        print(f"LEDGER VIOLATION: {exc}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_plan(args) -> int:
    from ..methods.base import make_method, plan_table

    cfg = loader.load(args.config, args.set)
    method = make_method(cfg)
    rows = plan_table([method])
    print(json.dumps({"config": loader.describe(cfg), "plan": rows}, indent=2))
    return 0


def cmd_run(args) -> int:
    from ..methods.base import make_method
    from ..training.runner import Runner

    cfg = loader.load(args.config, args.set)
    out = Path(cfg.runtime.out_dir) / f"{cfg.method}-s{cfg.seed}"
    ledger, out = _prepare(cfg, out)
    method = make_method(cfg)
    runner = Runner(cfg, method.plan(), ledger)

    results = []
    # Shared audit_seed so P_audit is the same question set across stages of
    # one run (see E1 pairing fix, §1.21); without it each stage's audit set
    # drifts with the target stage and cross-stage P_audit is unpaired.
    audit_seed = cfg.exp.audit_seed
    for stage in cfg.stages_to_run:
        runner.warmup_judge(stage)
        ctx = runner.make_context(stage, n_prompts=cfg.data.n_prompts_per_stage,
                                 seed=100 * stage,
                                 audit_stage=stage - 1 if stage > 1 else None,
                                 audit_seed=audit_seed)
        for step in range(1, cfg.train.max_steps_per_stage + 1):
            runner.train_step(ctx, step)
            if step % cfg.runtime.log_every == 0:
                obs = runner.measure_onpolicy(ctx, step)
                results.append({"stage": stage, "step": step, **obs.values})
    return _finish(ledger, out, {"rows": results},
                   allow_gold_control=cfg.is_reference or cfg.is_diagnostic_arm)


def cmd_e1(args) -> int:
    from ..experiments.e1_switch import run_e1, run_e1_prime

    cfg = loader.load(args.config, args.set)
    out = Path(cfg.runtime.out_dir) / f"e1-s{cfg.seed}"
    ledger, out = _prepare(cfg, out)
    main = run_e1(cfg, ledger, steps=args.steps,
                 seeds=args.seeds or cfg.exp.e1_seeds,
                 gold_seeds=cfg.exp.e1_gold_control_seeds)
    payload = {"e1": main.as_dict()}
    if args.heldout:
        prime = run_e1_prime(cfg, ledger, steps=args.steps)
        payload["e1_prime"] = prime.as_dict()
    return _finish(ledger, out, payload, allow_gold_control=True)


def cmd_e2(args) -> int:
    from ..experiments.e2_intervention import run_e2

    cfg = loader.load(args.config, args.set)
    out = Path(cfg.runtime.out_dir) / f"e2-s{cfg.seed}"
    ledger, out = _prepare(cfg, out)
    res = run_e2(cfg, ledger, stage=args.stage, audit_stage=args.audit_stage,
                 protocol_b=not args.no_protocol_b,
                 k_steps=args.k or (cfg.exp.e2_protocol_b_k[0]
                                    if cfg.exp.e2_protocol_b_k else 20))
    return _finish(ledger, out, res.as_dict(), allow_gold_control=True)


def cmd_ledger(args) -> int:
    path = Path(args.path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    gold_training = [r for r in rows if r["level"] == "L3" and r["in_training_scope"]]
    denied = [r for r in rows if not r["granted"]]
    print(json.dumps({
        "path": str(path),
        "records": len(rows),
        "L3_in_training": len(gold_training),
        "denied": len(denied),
        "by_level": {lvl: sum(1 for r in rows if r["level"] == lvl)
                     for lvl in ("L1", "L2", "L3")},
    }, indent=2))
    return 0


def cmd_e3(args) -> int:
    from ..experiments.e3_methods import run_e3

    cfg = loader.load(args.config, args.set)
    out = Path(cfg.runtime.out_dir) / f"e3-s{cfg.seed}"
    ledger, out = _prepare(cfg, out)
    res = run_e3(cfg, ledger, stages=tuple(args.stages),
                 steps_per_stage=args.steps, seed=cfg.seed,
                 n_seeds=args.n_seeds)
    return _finish(ledger, out, res.as_dict(), allow_gold_control=False)


def cmd_selftest(args) -> int:
    import pytest

    return int(pytest.main(["tests", "-x", "-q", "--disable-warnings"]))


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="csr_grpo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", required=True, help="path to a YAML preset")
        sp.add_argument("--set", nargs="*", default=[],
                        help="dotted overrides, e.g. train.lr=3e-5")

    sp = sub.add_parser("plan", help="print the method plan")
    common(sp)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run", help="train one method")
    common(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("e1", help="E1 switch-vs-continue with gold DID")
    common(sp)
    sp.add_argument("--steps", type=int, default=60)
    sp.add_argument("--seeds", type=int, default=1)
    sp.add_argument("--heldout", action="store_true",
                    help="also run E1' on the held-out Stage 2->3 pair")
    sp.set_defaults(func=cmd_e1)

    sp = sub.add_parser("e2", help="E2 reward-source intervention")
    common(sp)
    sp.add_argument("--stage", type=int, default=2)
    sp.add_argument("--audit-stage", type=int, default=1)
    sp.add_argument("--k", type=int, default=20)
    sp.add_argument("--no-protocol-b", action="store_true")
    sp.set_defaults(func=cmd_e2)

    sp = sub.add_parser("e3", help="E3 method-benefit gate")
    common(sp)
    sp.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    sp.add_argument("--steps", type=int, default=60)
    sp.add_argument("--n-seeds", type=int, default=1)
    sp.set_defaults(func=cmd_e3)

    sp = sub.add_parser("ledger", help="summarise a ledger.jsonl")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_ledger)

    sp = sub.add_parser("selftest", help="run the test suite")
    sp.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
