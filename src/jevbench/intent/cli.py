"""`jevbench intent ...`: Experiment 1, Jev vs an LLM for e-commerce intent classification."""

from __future__ import annotations

import argparse
import asyncio
import sys

from jevbench.common.cli_util import display, refuse_dirty_full_run, resolve_run_dir, run_id_arg


def cmd_prepare(_: argparse.Namespace) -> None:
    from jevbench.intent.data import build_splits, load_raw, write_splits
    from jevbench.intent.intents import load_catalog

    catalog = load_catalog()
    splits = build_splits(load_raw(), catalog)
    for name, path in write_splits(splits).items():
        frame = splits[name]
        profane, typos = frame["has_profanity"].mean(), frame["has_typos"].mean()
        print(
            f"{name:10s} {len(frame):5d} rows  intents={frame['intent'].nunique():2d}  "
            f"profane={profane:.0%}  typos={typos:.0%}  -> {display(path)}"
        )


def cmd_systems(_: argparse.Namespace) -> None:
    from jevbench.intent.classifiers import SYSTEMS

    for name, description in SYSTEMS.items():
        print(f"{name:14s} {description}")


async def _check(systems: list[str]) -> int:
    from jevbench.common import settings as settings_module
    from jevbench.intent import classifiers
    from jevbench.intent.intents import load_catalog

    settings, catalog = settings_module.load_settings(), load_catalog()
    failures = 0
    for name in systems:
        classifier = classifiers.build_classifier(name, settings, catalog)
        try:
            p = await classifier.classify("where is my order? it should have arrived yesterday")
        finally:
            await classifier.aclose()
        status = "OK " if p.outcome == "ok" else "ERR"
        failures += p.outcome != "ok"
        print(
            f"[{status}] {name:14s} {p.latency_ms:6.0f} ms  intent={p.intent}  conf={p.confidence}  "
            f"tokens in/out/reasoning={p.input_tokens}/{p.output_tokens}/{p.reasoning_tokens}  "
            f"model={p.model_reported}  cache_hit={p.cache_hit}  {p.error or ''}"
        )
    return failures


def cmd_check(args: argparse.Namespace) -> None:
    sys.exit(1 if asyncio.run(_check(args.systems)) else 0)


async def _run(args: argparse.Namespace) -> None:
    from jevbench.common import settings as settings_module
    from jevbench.common.runner import new_run_id
    from jevbench.common.tracing import setup_tracing, shutdown_tracing
    from jevbench.intent import classifiers, runner
    from jevbench.intent.intents import load_catalog

    refuse_dirty_full_run(args.limit, args.allow_dirty)

    settings, catalog = settings_module.load_settings(), load_catalog()
    run_id = args.run_id or new_run_id()
    project = setup_tracing(settings, args.system)
    print(f"run_id={run_id}  system={args.system}  phoenix_project={project or 'disabled'}")

    classifier = classifiers.build_classifier(args.system, settings, catalog)
    try:
        for repeat in args.repeats:
            target = runner.target(run_id, args.system, args.split, repeat)
            await runner.run_target(target, classifier, settings, catalog, limit=args.limit, warmup=args.warmup)
    finally:
        await classifier.aclose()
        shutdown_tracing()
    print(f"\nResults: results/intent/{run_id}/   Next: jevbench intent report --run-id {run_id}")


def cmd_run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


def _run_dir(run_id: str | None):
    from jevbench.intent import runner

    return resolve_run_dir(runner.RESULTS_DIR, run_id, "jevbench intent run")


def cmd_report(args: argparse.Namespace) -> None:
    from jevbench.intent.report import ReportError, build_report

    run_dir = _run_dir(args.run_id)
    try:
        report = build_report(run_dir, allow_partial=args.allow_partial)
    except ReportError as exc:
        raise SystemExit(str(exc)) from None
    out = run_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {display(out)}")


def cmd_phoenix_sync(args: argparse.Namespace) -> None:
    from jevbench.common import settings as settings_module
    from jevbench.intent import phoenix_sync

    settings = settings_module.load_settings(require_credentials=False)
    run_dir = _run_dir(args.run_id)
    for target in phoenix_sync.targets_in(run_dir):
        if args.systems and target.system not in args.systems:
            continue
        url = phoenix_sync.log_experiment(target, settings.phoenix_base_url)
        print(f"{target.system}/{target.split}/r{target.repeat}: {url}")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    from jevbench.intent.classifiers import SYSTEMS

    parser = sub.add_parser("intent", help="Experiment 1: intent classification on Bitext Retail eCommerce.")
    commands = parser.add_subparsers(dest="intent_command", required=True)

    commands.add_parser("prepare", help="Build data/intent/splits/*.jsonl from the raw Bitext CSV.").set_defaults(
        func=cmd_prepare
    )
    commands.add_parser("systems", help="List the systems under test.").set_defaults(func=cmd_systems)

    check = commands.add_parser("check", help="One live classification per system, to validate credentials.")
    check.add_argument("--systems", nargs="+", default=["jev", "baseline"], choices=list(SYSTEMS))
    check.set_defaults(func=cmd_check)

    run = commands.add_parser("run", help="Run one system over one split, sequentially.")
    run.add_argument("--system", required=True, choices=list(SYSTEMS))
    run.add_argument("--split", default="main", choices=["practice", "main", "profanity"])
    run.add_argument("--repeats", nargs="+", type=int, default=[1], help="Repeat numbers to run, e.g. 1 2 3.")
    run.add_argument("--limit", type=int, default=None, help="Only the first N examples (smoke tests).")
    run.add_argument("--warmup", type=int, default=20, help="Discarded warm-up calls before timing (0 to skip).")
    run.add_argument(
        "--run-id", type=run_id_arg, default=None, help="Group several systems under one run (default: timestamp)."
    )
    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a full run from uncommitted code (scratch runs only: the results will not trace to a commit).",
    )
    run.set_defaults(func=cmd_run)

    report = commands.add_parser("report", help="Write results/intent/<run>/report.md (no API keys needed).")
    report.add_argument("--run-id", type=run_id_arg, default=None, help="Default: the most recent run.")
    report.add_argument(
        "--allow-partial", action="store_true", help="Report even if systems ran different examples (with a warning)."
    )
    report.set_defaults(func=cmd_report)

    sync = commands.add_parser("phoenix-sync", help="Log a finished run into Phoenix as datasets + experiments.")
    sync.add_argument("--run-id", type=run_id_arg, default=None, help="Default: the most recent run.")
    sync.add_argument("--systems", nargs="+", choices=list(SYSTEMS), help="Only these systems (default: all).")
    sync.set_defaults(func=cmd_phoenix_sync)
