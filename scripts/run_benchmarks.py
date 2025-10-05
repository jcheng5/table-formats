#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

FORMAT_KEYS: List[str] = [
    "json",
    "csv",
    "xml",
    "yaml",
    "html",
    "markdown_table",
    "markdown_kv",
    "ini",
    "pipe_delimited",
    "jsonl",
    "natural_language",
]


def resolve_inspect_bin() -> str:
    env_override = os.environ.get("INSPECT_BIN")
    if env_override:
        return env_override

    candidates = [
        Path(sys.prefix) / "bin" / "inspect",
        Path(sys.prefix) / "Scripts" / "inspect.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "inspect"


def sanitize(segment: str) -> str:
    return segment.replace("/", "_").replace(":", "_").replace(" ", "_")


def run_evaluations(
    models: Iterable[str],
    formats: Iterable[str],
    limit: int | None,
    log_root: Path | None,
    extra_args: Iterable[str],
) -> None:
    inspect_bin = resolve_inspect_bin()

    for model in models:
        model_label = sanitize(model)
        for fmt in formats:
            task_name = f"evals/table_formats_eval.py@table_formats_{fmt}"
            print(f"\n→ Running {fmt} format on {model}...")

            cmd = [inspect_bin, "eval", task_name, "--model", model]
            if limit is not None:
                cmd.extend(["--limit", str(limit)])

            log_dir = None
            if log_root is not None:
                log_dir = log_root / model_label / fmt
                log_dir.mkdir(parents=True, exist_ok=True)
                cmd.extend(["--log-dir", str(log_dir)])

            cmd.extend(extra_args)

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"✖ Eval failed for format={fmt}, model={model} (exit code {exc.returncode})")
                if log_dir is not None:
                    print(f"  Inspect logs: {log_dir}")
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run table format benchmarks with Inspect")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Fully qualified Inspect model identifiers to evaluate (e.g. openai/gpt-4.1-mini)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=FORMAT_KEYS,
        default=FORMAT_KEYS,
        help="Subset of formats to evaluate (defaults to all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of samples to run for quick smoke tests",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("inspect-logs"),
        help="Directory where Inspect logs should be written (default: inspect-logs)",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable per-run log directories",
    )
    parser.add_argument(
        "--inspect-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed through to `inspect eval`",
    )

    args = parser.parse_args()

    log_root = None if args.no_logs else args.log_dir

    if log_root is not None:
        log_root.mkdir(parents=True, exist_ok=True)

    run_evaluations(
        models=args.models,
        formats=args.formats,
        limit=args.limit,
        log_root=log_root,
        extra_args=args.inspect_args,
    )


if __name__ == "__main__":
    main()
