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
    "fleurs": ("transcription", "raw_transcription", "sentence"),
}

# Audio column name per source — most datasets call it "audio", but IndicVoices
# names its audio-typed column "audio_filepath". Confirm against
# 01_explore_datasets.py output before a full run, same as TRANSCRIPT_KEYS.
AUDIO_KEYS: dict[str, str] = {
    "medibeng": "audio",
    "indicvoices": "audio_filepath",
    "fleurs": "audio",
}

SOURCE_SPECS = {
    "medibeng": {"path": "pr0mila-gh0sh/MediBeng", "split": "train"},
    # trust_remote_code=True runs each repo's loading script.
    # split="valid" not "train" — train is 745GB across ~96 multi-GB shards,
    # and streaming still downloads a whole shard before yielding any rows.
    # valid is far smaller and plenty since 04_build_dataset.py reshuffles
    # everything into fresh train/val/test splits downstream anyway.
    "indicvoices": {
        "path": "ai4bharat/IndicVoices",
        "name": "bengali",
        "split": "valid",
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
    audio_key = AUDIO_KEYS[source]
    ds = load_dataset(**spec, streaming=streaming)
    ds = ds.cast_column(audio_key, Audio(sampling_rate=TARGET_SR))
    iterator = iter(ds)

    kept, dropped, rows = 0, 0, []
    for sample in iterator:
        if limit is not None and kept >= limit:
            break
        if sample.get(audio_key) is None or sample[audio_key].get("array") is None:
            dropped += 1
            continue
        audio = resample_audio(sample[audio_key]["array"], sample[audio_key]["sampling_rate"])
        sentence = pick_transcript(sample, source)
        if not is_valid_sample(audio, sentence):
            dropped += 1
            continue
        rows.append(unified(audio, sentence, source))
        kept += 1

    print(f"{source}: kept {kept}, dropped {dropped}")
    if not rows:
        raise SystemExit(
            f"{source}: 0 samples kept out of {dropped} seen — every row was rejected "
            "(missing/null audio or invalid duration/transcript). Run "
            "01_explore_datasets.py to inspect the raw fields before retrying."
        )
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
