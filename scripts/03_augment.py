"""Audio augmentation to synthetically expand speaker diversity.

Applied to code-switched / in-domain audio only (MediBeng, real meeting recordings).
IndicVoices and Common Voice are already speaker-diverse — augmenting them adds
noise without adding coverage.

Usage:
    python scripts/03_augment.py --source medibeng --n-augments 4
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from audiomentations import AddGaussianNoise, Compose, PitchShift, TimeStretch
from datasets import Audio, Dataset, load_from_disk

from common import REPO_ROOT, TARGET_SR, load_yaml, unified

AUGMENTED_DIR = REPO_ROOT / "data" / "augmented"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def build_pipeline(cfg: dict) -> Compose:
    aug = cfg["augmentation"]
    return Compose(
        [
            PitchShift(**aug["pitch_shift"]),
            TimeStretch(**aug["time_stretch"]),
            AddGaussianNoise(**aug["gaussian_noise"]),
        ]
    )


def augment_sample(
    pipeline: Compose,
    audio_array: np.ndarray,
    sample_rate: int = TARGET_SR,
    n_augments: int = 4,
) -> list[np.ndarray]:
    """Generate N augmented versions of one audio sample."""
    return [
        pipeline(samples=audio_array.astype(np.float32), sample_rate=sample_rate)
        for _ in range(n_augments)
    ]


def main() -> None:
    cfg = load_yaml("training_config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="medibeng")
    parser.add_argument("--n-augments", type=int, default=cfg["augmentation"]["n_augments"])
    parser.add_argument("--include-original", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=cfg["common"]["seed"])
    args = parser.parse_args()

    # audiomentations draws from the global RNGs; seeding keeps this stage
    # reproducible so DVC's cache is not invalidated on every run.
    random.seed(args.seed)
    np.random.seed(args.seed)

    allowed = cfg["augmentation"]["sources"]
    if args.source not in allowed:
        raise SystemExit(f"{args.source} is not in augmentation.sources ({allowed})")

    src: Path = PROCESSED_DIR / args.source
    ds = load_from_disk(str(src))
    pipeline = build_pipeline(cfg)

    rows = []
    for sample in ds:
        audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
        if args.include_original:
            rows.append(unified(audio, sample["sentence"], sample["source"]))
        for variant in augment_sample(pipeline, audio, TARGET_SR, args.n_augments):
            rows.append(unified(variant, sample["sentence"], f"{sample['source']}_aug"))

    out = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=TARGET_SR))
    dest = AUGMENTED_DIR / args.source
    out.save_to_disk(str(dest))
    print(f"{len(ds)} → {len(out)} samples  saved → {dest}")


if __name__ == "__main__":
    main()
