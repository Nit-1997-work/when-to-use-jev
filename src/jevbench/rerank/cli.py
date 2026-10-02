"""`jevbench rerank ...`: Experiment 2, Jev vs an LLM for product search re-ranking."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from jevbench.common.cli_util import display, refuse_dirty_full_run, resolve_run_dir, run_id_arg

SPLITS = ("practice", "practice-nomatch", "main", "nomatch", "consistency", "reversed")


def cmd_prepare(args: argparse.Namespace) -> None:
    from jevbench.rerank import data

    frame = data.load_esci()
    if args.grocery_candidates:
        pool = data.grocery_candidates(frame, data.load_grocery_terms())
        path = data.write_grocery_review(pool)
        print(f"{len(pool)} candidate searches -> {display(path)}. Set `grocery` on each, then run prepare again.")
        return
    reviewed, approved = data.load_grocery_review()
    splits = data.build_splits(frame, reviewed=reviewed, approved=approved)
    paths = data.write_splits(splits)
    for line, path in zip(data.profile(splits), paths.values(), strict=True):
        print(f"{line}  -> {display(path)}")


def cmd_systems(_: argparse.Namespace) -> None:
    from jevbench.rerank.systems import SYSTEMS

    for name, description in SYSTEMS.items():
        print(f"{name:22s} {description}")


async def _check(systems: list[str]) -> int:
    from jevbench.common import settings as settings_module
    from jevbench.rerank import systems as rerank_systems
    from jevbench.rerank.data import load_split
    from jevbench.rerank.labels import load_labels

    settings, labels = settings_module.load_settings(), load_labels()
    # The practice search with the most candidates: checks request size limits as well as credentials.
    search = max(load_split("practice"), key=lambda s: (len(s.candidates), s.id))
    print(f"search {search.id}: {search.query!r}, {len(search.candidates)} candidates")
    failures = 0
    for name in systems:
        reranker = rerank_systems.build_reranker(name, settings, labels)
        try:
            r = await reranker.rerank(search)
        except Exception as exc:  # a check reports every failure, retryable or not
            failures += 1
            print(f"[ERR] {name:22s} {type(exc).__name__}: {exc}")
            continue
        finally:
            await reranker.aclose()
        ok = r.outcome == "ok"
        failures += not ok
        top = ",".join(r.order[:3])
        print(
            f"[{'OK ' if ok else 'ERR'}] {name:22s} {r.latency_ms:7.0f} ms  calls={r.calls}  top3={top}  "
            f"tokens in/out/reasoning={r.input_tokens}/{r.output_tokens}/{r.reasoning_tokens}  "
            f"model={r.model_reported}  ids invented/duplicate/missing={r.invented_ids}/{r.duplicate_ids}/"
            f"{r.missing_ids}  blocked calls={r.blocked_calls}  {r.error or ''}"
        )
    return failures


def cmd_check(args: argparse.Namespace) -> None:
    sys.exit(1 if asyncio.run(_check(args.systems)) else 0)


async def _run(args: argparse.Namespace) -> None:
    from jevbench.common import settings as settings_module
    from jevbench.common.runner import new_run_id
    from jevbench.common.tracing import setup_tracing, shutdown_tracing
    from jevbench.rerank import runner
    from jevbench.rerank import systems as rerank_systems
    from jevbench.rerank.labels import load_labels

    refuse_dirty_full_run(args.limit, args.allow_dirty)

    settings, labels = settings_module.load_settings(), load_labels()
    run_id = args.run_id or new_run_id()
    project = setup_tracing(settings, f"rerank-{args.system}")
    print(f"run_id={run_id}  system={args.system}  split={args.split}  phoenix_project={project or 'disabled'}")

    reranker = rerank_systems.build_reranker(args.system, settings, labels)
    try:
        for repeat in args.repeats:
            target = runner.target(run_id, args.system, args.split, repeat)
            await runner.run_target(target, reranker, settings, labels, limit=args.limit, warmup=args.warmup)
    finally:
        await reranker.aclose()
        shutdown_tracing()
    print(f"\nResults: results/rerank/{run_id}/   Next: jevbench rerank report --run-id {run_id}")


def cmd_run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


def _run_dir(run_id: str | None) -> Path:
    from jevbench.rerank import runner

    return resolve_run_dir(runner.RESULTS_DIR, run_id, "jevbench rerank run")


def cmd_report(args: argparse.Namespace) -> None:
    from jevbench.rerank.report import ReportError, build_report

    run_dir = _run_dir(args.run_id)
    try:
        report = build_report(run_dir, allow_partial=args.allow_partial)
    except ReportError as exc:
        raise SystemExit(str(exc)) from None
    out = run_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {display(out)}")


def cmd_tune_thresholds(args: argparse.Namespace) -> None:
    from jevbench.common.stats import answered, load_results
    from jevbench.rerank.metrics import tune_threshold
    from jevbench.rerank.thresholds import write_thresholds

    run_dir = _run_dir(args.run_id)
    frame = load_results(run_dir)
    practice = frame[frame["split"].isin(["practice", "practice-nomatch"]) & (frame["repeat"] == 1)]
    if practice.empty:
        raise SystemExit(f"{display(run_dir)} has no practice or practice-nomatch records to tune on.")
    tuned: dict[str, tuple[float, float]] = {}
    for system, part in practice.groupby("system", sort=True):
        if part["abstain_flag"].notna().any():
            print(f"{system:22s} decides by its own flag; no threshold")
            continue
        best = tune_threshold(part[answered(part)])
        if best is None:
            print(f"{system:22s} gives no confidence to threshold")
            continue
        tuned[str(system)] = best
        print(f"{system:22s} threshold={best[0]:.4f}  practice F1={best[1]:.3f}")
    path = write_thresholds(tuned, run_dir.name)
    print(f"Written to {display(path)}. Freeze it (commit) before the full run.")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    from jevbench.rerank.systems import SYSTEMS

    parser = sub.add_parser("rerank", help="Experiment 2: product search re-ranking on Amazon ESCI.")
    commands = parser.add_subparsers(dest="rerank_command", required=True)

    prepare = commands.add_parser("prepare", help="Build data/rerank/splits/*.jsonl from the pinned ESCI files.")
    prepare.add_argument(
        "--grocery-candidates",
        action="store_true",
        help="Instead, write the candidate searches for the grocery review to data/rerank/grocery_review.yaml.",
    )
    prepare.set_defaults(func=cmd_prepare)

    commands.add_parser("systems", help="List the systems under test.").set_defaults(func=cmd_systems)

    check = commands.add_parser("check", help="One live search per system, to validate credentials and requests.")
    check.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    check.set_defaults(func=cmd_check)

    run = commands.add_parser("run", help="Run one system over one split, one search at a time.")
    run.add_argument("--system", required=True, choices=list(SYSTEMS))
    run.add_argument("--split", default="main", choices=list(SPLITS))
    run.add_argument("--repeats", nargs="+", type=int, default=[1], help="Repeat numbers to run, e.g. 1 2 3.")
    run.add_argument("--limit", type=int, default=None, help="Only the first N searches (smoke tests).")
    run.add_argument("--warmup", type=int, default=20, help="Discarded warm-up searches before timing (0 to skip).")
    run.add_argument(
        "--run-id", type=run_id_arg, default=None, help="Group several systems under one run (default: timestamp)."
    )
    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a full run from uncommitted code (scratch runs only: the results will not trace to a commit).",
    )
    run.set_defaults(func=cmd_run)

    report = commands.add_parser("report", help="Write results/rerank/<run>/report.md (no API keys needed).")
    report.add_argument("--run-id", type=run_id_arg, default=None, help="Default: the most recent run.")
    report.add_argument(
        "--allow-partial", action="store_true", help="Report even if systems ran different searches (with a warning)."
    )
    report.set_defaults(func=cmd_report)

    tune = commands.add_parser(
        "tune-thresholds", help="Pick each system's abstention threshold from a practice run (writes thresholds.yaml)."
    )
    tune.add_argument("--run-id", type=run_id_arg, required=True, help="A run over practice + practice-nomatch.")
    tune.set_defaults(func=cmd_tune_thresholds)
