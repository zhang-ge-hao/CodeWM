"""Command-line interface for single-file obfuscation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .engine import RandomWalkObfuscator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Python source file")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="default: stdout")
    parser.add_argument("--trace", type=Path, help="optional JSON trace")
    parser.add_argument(
        "--list-actions",
        action="store_true",
        help="print current concrete action counts without walking",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.input.read_text(encoding="utf-8")
    engine = RandomWalkObfuscator(source, seed=args.seed)
    if args.list_actions:
        json.dump(engine.action_counts(source), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    result = engine.walk(source, args.steps)
    if args.output is None:
        sys.stdout.write(result.source)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.source, encoding="utf-8")
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "steps": args.steps,
                    "rule_counts": result.rule_counts,
                    "trace": [record.to_dict() for record in result.records],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
