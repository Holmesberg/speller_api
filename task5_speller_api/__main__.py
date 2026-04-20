"""CLI entry point — lets teammates smoke-test without writing Python.

Example:

    python -m task5_speller_api he --context "writing an email to my professor"
    → ["hello", "hope", "help"]
"""

from __future__ import annotations

import argparse
import json
import sys

from .speller import predict_words


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m task5_speller_api",
        description="Predict three complete words from a BCI-emitted prefix.",
    )
    parser.add_argument("prefix", help="letters committed so far (usually 1-3)")
    parser.add_argument(
        "--context",
        default="",
        help="Optional scenario context, e.g. 'ordering food at a restaurant'",
    )
    parser.add_argument(
        "--sentence",
        default="",
        help="Optional sentence being composed up to the current word",
    )
    args = parser.parse_args(argv)

    try:
        words = predict_words(
            prefix=args.prefix, context=args.context, sentence=args.sentence
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(words))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
