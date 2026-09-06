"""Validate the versioned RAG evaluation dataset before a RAGAS run.

RAGAS requires a configured model provider and a populated document corpus, so this
entry point deliberately checks the dataset contract locally instead of claiming a
model-based score without a reproducible environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = {"question", "ground_truth"}
DEFAULT_DATASET = Path("evals/datasets/ragas_dataset.jsonl")


def validate_dataset(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    valid_rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Row {line_number} is missing: {', '.join(sorted(missing))}")
        valid_rows += 1

    if not valid_rows:
        raise ValueError("Evaluation dataset contains no examples.")
    return valid_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--check", action="store_true", help="Validate only (default).")
    args = parser.parse_args()

    rows = validate_dataset(args.dataset)
    print(f"Validated {rows} RAG evaluation examples in {args.dataset}.")
    print("Run model-backed RAGAS evaluation after configuring an LLM and corpus.")


if __name__ == "__main__":
    main()
