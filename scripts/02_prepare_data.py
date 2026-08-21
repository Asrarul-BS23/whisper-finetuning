"""Normalize every source into one schema.

Output schema: {"audio": {"array": float32 @16k, "sampling_rate": 16000},
                "sentence": str, "source": str}

Rules: 16 kHz mono, 1s <= duration <= 30s, cleaned transcript, drop the rest.

TRANSCRIPT COLUMN NAMES ARE PER-SOURCE GUESSES — confirm them against
`01_explore_datasets.py` output and fix TRANSCRIPT_KEYS before a full run.

Usage:
    python scripts/02_prepare_data.py --source medibeng --limit 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Audio, Dataset, load_dataset

from common import (
    REPO_ROOT,
    TARGET_SR,
    clean_transcript,
    is_valid_sample,
    resample_audio,
    unified,
)

# Candidate transcript columns, tried in order — first present wins.
TRANSCRIPT_KEYS: dict[str, tuple[str, ...]] = {
    "medibeng": ("sentence", "transcription", "text"),
    "indicvoices": ("text", "transcript", "sentence"),
    "common_voice": ("sentence",),
    "fleurs": ("transcription", "raw_transcription", "sentence"),
}

SOURCE_SPECS = {
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

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def pick_transcript(sample: dict, source: str) -> str:
    for key in TRANSCRIPT_KEYS[source]:
        if key in sample and sample[key]:
            return clean_transcript(str(sample[key]))
    raise KeyError(
        f"no transcript column for {source}; saw {sorted(sample)} — "
        f"update TRANSCRIPT_KEYS after running 01_explore_datasets.py"
    )


def process(source: str, limit: int | None, streaming: bool) -> Dataset:
    spec = SOURCE_SPECS[source]
    ds = load_dataset(**spec, streaming=streaming)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    iterator = iter(ds)

    kept, dropped, rows = 0, 0, []
    for sample in iterator:
        if limit is not None and kept >= limit:
            break
        audio = resample_audio(sample["audio"]["array"], sample["audio"]["sampling_rate"])
        sentence = pick_transcript(sample, source)
        if not is_valid_sample(audio, sentence):
            dropped += 1
            continue
        rows.append(unified(audio, sentence, source))
        kept += 1

    print(f"{source}: kept {kept}, dropped {dropped}")
    return Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=TARGET_SR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCE_SPECS), required=True)
    parser.add_argument("--limit", type=int, default=None, help="cap kept samples (debug)")
    parser.add_argument("--no-streaming", action="store_true")
    args = parser.parse_args()

    ds = process(args.source, args.limit, streaming=not args.no_streaming)
    out: Path = PROCESSED_DIR / args.source
    ds.save_to_disk(str(out))
    print(f"saved {len(ds)} samples → {out}")


if __name__ == "__main__":
    main()
