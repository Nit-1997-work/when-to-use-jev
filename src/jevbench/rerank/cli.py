"""`jevbench rerank ...`: Experiment 2, Jev vs an LLM for product search re-ranking."""

from __future__ import annotations

import argparse


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    parser = sub.add_parser("rerank", help="Experiment 2: product search re-ranking on Amazon ESCI.")
    parser.add_subparsers(dest="rerank_command", required=True)
