"""CLI entry point — lets teammates smoke-test without writing Python.

Example:

    python -m task5_speller_api he --context "writing an email to my professor"
    → ["hello", "hope", "help"]
"""

from __future__ import annotations

import argparse
import json
import sys

from .speller import API


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m task5_speller_api",
        description="Predict three complete words from a 1-3 letter prefix.",
    )
    parser.add_argument("prefix", help="1-3 alphabetic characters")
    parser.add_argument(
        "--context",
        default="",
        help="Optional scenario context, e.g. 'ordering food at a restaurant'",
    )
    args = parser.parse_args(argv)

    try:
        api = API()
        words = api.predict_words(context=args.context, prefix=args.prefix, sentence="")
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
