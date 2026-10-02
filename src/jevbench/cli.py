"""`jevbench` command line. One subcommand group per experiment, plus shared tools.

jevbench intent ...    Experiment 1: intent classification (prepare, systems, check, run, report, phoenix-sync)
jevbench rerank ...    Experiment 2: product search re-ranking (prepare, systems, check, run, report, ...)
jevbench audit         Pre-publish scan for the gateway host, API keys, and .audit-denylist patterns
jevbench phoenix-setup Register Jev's token price in Phoenix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jevbench.common.cli_util import use_system_trust_store
from jevbench.common.paths import REPO_ROOT
from jevbench.common.settings import load_env


def cmd_audit(args: argparse.Namespace) -> None:
    from jevbench.common.audit import committable_files, deny_rules, expand, scan

    files = expand([Path(p) for p in args.paths]) if args.paths else committable_files(REPO_ROOT)
    findings = scan(files, deny_rules(REPO_ROOT))
    for finding in findings:
        print(finding)
    print(f"audit: {len(files)} files scanned, {len(findings)} finding(s)")
    sys.exit(1 if findings else 0)


def cmd_phoenix_setup(_: argparse.Namespace) -> None:
    from jevbench.common import settings as settings_module
    from jevbench.intent.phoenix_sync import register_jev_price

    settings = settings_module.load_settings(require_credentials=False)
    if not settings.phoenix_base_url:
        raise SystemExit("Set PHOENIX_BASE_URL in .env.")
    print(register_jev_price(settings.phoenix_base_url, settings.typesafe_model, settings.jev_input_price))


def build_parser() -> argparse.ArgumentParser:
    from jevbench.intent import cli as intent_cli
    from jevbench.rerank import cli as rerank_cli

    parser = argparse.ArgumentParser(
        prog="jevbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    intent_cli.add_parser(sub)
    rerank_cli.add_parser(sub)

    audit = sub.add_parser(
        "audit", help="Scan files before publishing for the gateway host, API keys, and .audit-denylist patterns."
    )
    audit.add_argument("paths", nargs="*", help="Files or directories to scan (default: every file git would commit).")
    audit.set_defaults(func=cmd_audit)

    setup = sub.add_parser("phoenix-setup", help="Register Jev's token price in Phoenix (run before a run).")
    setup.set_defaults(func=cmd_phoenix_setup)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_env()  # first, so settings in .env (including JEVBENCH_TRUSTSTORE) apply to every command
    use_system_trust_store()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
