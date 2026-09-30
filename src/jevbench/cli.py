"""`jevbench` command line: prepare data, run systems, report, audit, and mirror results into Phoenix."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

from jevbench.settings import REPO_ROOT, load_env

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def run_id_arg(value: str) -> str:
    """Run ids become directory names under results/, so they must be a single safe path segment."""
    if not _RUN_ID.match(value) or ".." in value:
        raise argparse.ArgumentTypeError("use letters, digits, '.', '_' or '-' (max 64 chars)")
    return value


def _use_system_trust_store() -> None:
    """Trust the OS certificate store (e.g. corporate TLS inspection roots) unless disabled."""
    if os.environ.get("JEVBENCH_TRUSTSTORE", "1").strip() != "0":
        import truststore

        truststore.inject_into_ssl()


def _display(path: Path) -> str:
    """A path relative to the repository when it is inside it."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _tree_is_clean() -> bool:
    """True only when git confirms there are no uncommitted changes; an unknown state does not count as clean."""
    from jevbench.runner import git_state

    dirty = git_state()["dirty"]
    return isinstance(dirty, bool) and not dirty


# ---------- commands ----------


def cmd_prepare(_: argparse.Namespace) -> None:
    from jevbench.data import build_splits, load_raw, write_splits
    from jevbench.intents import load_catalog

    catalog = load_catalog()
    splits = build_splits(load_raw(), catalog)
    for name, path in write_splits(splits).items():
        frame = splits[name]
        profane, typos = frame["has_profanity"].mean(), frame["has_typos"].mean()
        print(
            f"{name:10s} {len(frame):5d} rows  intents={frame['intent'].nunique():2d}  "
            f"profane={profane:.0%}  typos={typos:.0%}  -> {_display(path)}"
        )


def cmd_systems(_: argparse.Namespace) -> None:
    from jevbench.classifiers import SYSTEMS

    for name, description in SYSTEMS.items():
        print(f"{name:14s} {description}")


async def _check(systems: list[str]) -> int:
    from jevbench.classifiers import build_classifier
    from jevbench.intents import load_catalog
    from jevbench.settings import load_settings

    settings, catalog = load_settings(), load_catalog()
    failures = 0
    for name in systems:
        classifier = build_classifier(name, settings, catalog)
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
    from jevbench.classifiers import build_classifier
    from jevbench.intents import load_catalog
    from jevbench.runner import RunTarget, new_run_id, run_target
    from jevbench.settings import load_settings
    from jevbench.tracing import setup_tracing, shutdown_tracing

    # Full runs produce publishable numbers, so they must trace to a commit. Smoke runs (--limit) are exempt.
    if args.limit is None and not args.allow_dirty and not _tree_is_clean():
        raise SystemExit(
            "Refusing a full run: the working tree has uncommitted changes (or git is unavailable), so the results "
            "would not trace to a commit. Commit first, or pass --allow-dirty for a scratch run."
        )

    settings, catalog = load_settings(), load_catalog()
    run_id = args.run_id or new_run_id()
    project = setup_tracing(settings, args.system)
    print(f"run_id={run_id}  system={args.system}  phoenix_project={project or 'disabled'}")

    classifier = build_classifier(args.system, settings, catalog)
    try:
        for repeat in args.repeats:
            target = RunTarget(run_id=run_id, system=args.system, split=args.split, repeat=repeat)
            await run_target(target, classifier, settings, catalog, limit=args.limit, warmup=args.warmup)
    finally:
        await classifier.aclose()
        shutdown_tracing()
    print(f"\nResults: results/{run_id}/   Next: jevbench report --run-id {run_id}")


def cmd_run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


def _run_dir(run_id: str | None) -> Path:
    from jevbench.runner import RESULTS_DIR, latest_run_dir

    if run_id:
        path = RESULTS_DIR / run_id
        if not path.is_dir():
            raise SystemExit(f"No such run: {path}")
        return path
    latest = latest_run_dir(RESULTS_DIR)
    if latest is None:
        raise SystemExit("No runs under results/. Run `jevbench run` first.")
    return latest


def cmd_report(args: argparse.Namespace) -> None:
    from jevbench.report import ReportError, build_report

    run_dir = _run_dir(args.run_id)
    try:
        report = build_report(run_dir, allow_partial=args.allow_partial)
    except ReportError as exc:
        raise SystemExit(str(exc)) from None
    out = run_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {_display(out)}")


def cmd_audit(args: argparse.Namespace) -> None:
    from jevbench.audit import committable_files, deny_rules, expand, scan

    files = expand([Path(p) for p in args.paths]) if args.paths else committable_files(REPO_ROOT)
    findings = scan(files, deny_rules(REPO_ROOT))
    for finding in findings:
        print(finding)
    print(f"audit: {len(files)} files scanned, {len(findings)} finding(s)")
    sys.exit(1 if findings else 0)


def cmd_phoenix_sync(args: argparse.Namespace) -> None:
    from jevbench.phoenix_sync import log_experiment, targets_in
    from jevbench.settings import load_settings

    settings = load_settings(require_credentials=False)
    run_dir = _run_dir(args.run_id)
    for target in targets_in(run_dir):
        if args.systems and target.system not in args.systems:
            continue
        url = log_experiment(target, settings.phoenix_base_url)
        print(f"{target.system}/{target.split}/r{target.repeat}: {url}")


def cmd_phoenix_setup(_: argparse.Namespace) -> None:
    from jevbench.phoenix_sync import register_jev_price
    from jevbench.settings import load_settings

    settings = load_settings(require_credentials=False)
    if not settings.phoenix_base_url:
        raise SystemExit("Set PHOENIX_BASE_URL in .env.")
    print(register_jev_price(settings.phoenix_base_url, settings.typesafe_model, settings.jev_input_price))


# ---------- parser ----------


def build_parser() -> argparse.ArgumentParser:
    from jevbench.classifiers import SYSTEMS

    parser = argparse.ArgumentParser(prog="jevbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Build data/splits/*.jsonl from the raw Bitext CSV.").set_defaults(func=cmd_prepare)
    sub.add_parser("systems", help="List the systems under test.").set_defaults(func=cmd_systems)

    check = sub.add_parser("check", help="One live classification per system, to validate credentials and wiring.")
    check.add_argument("--systems", nargs="+", default=["jev", "baseline"], choices=list(SYSTEMS))
    check.set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="Run one system over one split, sequentially.")
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

    report = sub.add_parser("report", help="Write results/<run>/report.md (no API keys needed).")
    report.add_argument("--run-id", type=run_id_arg, default=None, help="Default: the most recent run.")
    report.add_argument(
        "--allow-partial", action="store_true", help="Report even if systems ran different examples (with a warning)."
    )
    report.set_defaults(func=cmd_report)

    audit = sub.add_parser(
        "audit", help="Scan files before publishing for the gateway host, API keys, and .audit-denylist patterns."
    )
    audit.add_argument("paths", nargs="*", help="Files or directories to scan (default: every file git would commit).")
    audit.set_defaults(func=cmd_audit)

    sync = sub.add_parser("phoenix-sync", help="Log a finished run into Phoenix as datasets + experiments.")
    sync.add_argument("--run-id", type=run_id_arg, default=None, help="Default: the most recent run.")
    sync.add_argument("--systems", nargs="+", choices=list(SYSTEMS), help="Only these systems (default: all).")
    sync.set_defaults(func=cmd_phoenix_sync)

    setup = sub.add_parser("phoenix-setup", help="Register Jev's token price in Phoenix (run before a run).")
    setup.set_defaults(func=cmd_phoenix_setup)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_env()  # first, so settings in .env (including JEVBENCH_TRUSTSTORE) apply to every command
    _use_system_trust_store()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
