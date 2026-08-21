"""Explore each source dataset before processing.

Run this first: column names, audio format and sample rates differ per source,
and Steps 2-4 hard-code those keys. Common Voice is gated — accept its terms on
the Hub and export HF_TOKEN first.

Usage:
    python scripts/01_explore_datasets.py                 # all sources, streaming
    python scripts/01_explore_datasets.py --only medibeng
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import load_dataset

SOURCES: dict[str, dict[str, Any]] = {
    "medibeng": {"path": "pr0mila-gh0sh/MediBeng", "split": "train"},
    "indicvoices": {
        "path": "ai4bharat/IndicVoices",
        "name": "bengali",
        "split": "train",
        "trust_remote_code": True,
    },
    "common_voice": {
        "path": "mozilla-foundation/common_voice_17_0",
        "name": "bn",
        "split": "train",
        "trust_remote_code": True,
    },
    "fleurs": {
        "path": "google/fleurs",
        "name": "bn_in",
        "split": "train",
        "trust_remote_code": True,
    },
}


def describe(key: str, spec: dict[str, Any], streaming: bool) -> None:
    print(f"\n{'=' * 70}\n{key}  ←  {spec['path']} {spec.get('name', '')}\n{'=' * 70}")
    try:
        ds = load_dataset(**spec, streaming=streaming)
    except Exception as exc:  # noqa: BLE001 — exploration should not abort the sweep
        print(f"  FAILED to load: {type(exc).__name__}: {exc}")
        return

    if streaming:
        sample = next(iter(ds))
        print("  columns:", list(sample.keys()))
        print("  size: unknown (streaming)")
    else:
        sample = ds[0]
        print("  columns:", ds.column_names)
        print("  size:", len(ds))

    for col, value in sample.items():
        if isinstance(value, dict) and "sampling_rate" in value:
            arr = value.get("array")
            print(
                f"  {col}: audio sr={value['sampling_rate']} "
                f"len={len(arr) if arr is not None else '?'} path={value.get('path')}"
            )
        else:
            preview = json.dumps(value, ensure_ascii=False, default=str)
            print(f"  {col}: {preview[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(SOURCES), nargs="*")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="download fully instead of streaming (gives exact sizes, costs disk)",
    )
    args = parser.parse_args()

    keys = args.only or list(SOURCES)
    for key in keys:
        describe(key, SOURCES[key], streaming=not args.no_streaming)


if __name__ == "__main__":
    main()
