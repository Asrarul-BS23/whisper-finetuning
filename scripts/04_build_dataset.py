"""Merge all processed sources into one DatasetDict, honoring the mix targets.

Composition target (configs/training_config.yaml → data.mix):
  MediBeng (+augmented) 20% | IndicVoices 40% | Common Voice 25% | FLEURS 15%

Sources are downsampled (never upsampled) to the largest mix that keeps every
ratio satisfiable, so the reported sizes may be smaller than what is on disk.

Usage:
    python scripts/04_build_dataset.py
    python scripts/04_build_dataset.py --push your-username/bangla-en-meeting-asr
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from common import REPO_ROOT, load_yaml

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
AUGMENTED_DIR = REPO_ROOT / "data" / "augmented"
FINAL_DIR = PROCESSED_DIR / "final"


def load_source(name: str) -> Dataset | None:
    """Prefer the augmented copy of a source when one exists."""
    for base in (AUGMENTED_DIR, PROCESSED_DIR):
        path: Path = base / name
        if (path / "dataset_info.json").exists():
            ds = load_from_disk(str(path))
            print(f"  {name}: {len(ds)} samples from {path.relative_to(REPO_ROOT)}")
            return ds
    print(f"  {name}: MISSING — run 02_prepare_data.py --source {name}")
    return None


def apply_mix(pools: dict[str, Dataset], mix: dict[str, float], seed: int) -> list[Dataset]:
    """Scale every pool down to the largest total that satisfies the ratios."""
    total = min(len(pools[k]) / mix[k] for k in pools if mix.get(k))
    parts = []
    for name, ds in pools.items():
        want = int(total * mix[name])
        part = ds.shuffle(seed=seed).select(range(min(want, len(ds))))
        print(f"  {name}: using {len(part)}/{len(ds)} (target {mix[name]:.0%})")
        parts.append(part)
    return parts


def build_final_dataset(parts: list[Dataset], seed: int) -> DatasetDict:
    combined = concatenate_datasets(parts).shuffle(seed=seed)
    split = combined.train_test_split(test_size=0.1, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": split["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )


def main() -> None:
    cfg = load_yaml("training_config.yaml")
    mix = cfg["data"]["mix"]
    seed = cfg["common"]["seed"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--push", default=cfg["data"]["hub_dataset_id"] or None)
    parser.add_argument("--no-mix", action="store_true", help="concatenate as-is, ignore ratios")
    args = parser.parse_args()

    print("loading sources:")
    pools = {name: ds for name in mix if (ds := load_source(name)) is not None}
    if not pools:
        raise SystemExit("no processed sources found — run 02_prepare_data.py first")

    print("applying mix:" if not args.no_mix else "skipping mix")
    parts = list(pools.values()) if args.no_mix else apply_mix(pools, mix, seed)

    final = build_final_dataset(parts, seed)
    final.save_to_disk(str(FINAL_DIR))
    print({k: len(v) for k, v in final.items()}, "→", FINAL_DIR)

    if args.push:
        final.push_to_hub(args.push, private=True)
        print(f"pushed → {args.push} (private)")


if __name__ == "__main__":
    main()
