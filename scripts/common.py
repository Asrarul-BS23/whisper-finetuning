"""Shared helpers: config loading, transcript cleaning, audio normalization, filtering."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

TARGET_SR = 16000
MIN_DURATION_S = 1
MAX_DURATION_S = 30

# Unified schema every processed sample must satisfy.
UNIFIED_COLUMNS = ("audio", "sentence", "source")


def load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Keys that live in a profile but belong to the model, not the trainer.
_MODEL_KEYS = ("base_model_id", "torch_dtype")


def load_configs(
    debug: bool, model_override: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (training_cfg, lora_cfg) with the debug/full profile merged in.

    The profile decides the checkpoint and the precision as well as the trainer
    settings: debug = whisper-small in fp16 on a T4, full = large-v3 in bf16 on
    the DGX. `model_override` swaps the checkpoint without touching anything else.
    """
    profile = "debug" if debug else "full"
    training = load_yaml("training_config.yaml")
    lora = load_yaml("lora_config.yaml")
    training_merged = {**training["common"], **training[profile]}
    lora_merged = {**lora["common"], **lora[profile]}

    model_cfg = {
        **training["model"],
        **{k: training_merged[k] for k in _MODEL_KEYS if k in training_merged},
    }
    if model_override:
        model_cfg["base_model_id"] = model_override

    return (
        {**training, "model": model_cfg, "profile": profile, "train": training_merged},
        lora_merged,
    )


def clean_transcript(text: str) -> str:
    """Remove noise tags, collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"\[.*?\]", "", text)  # [noise], [laughter]
    text = re.sub(r"\(.*?\)", "", text)  # (inaudible)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resample_audio(audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
    """Downmix to mono and resample to 16 kHz float32."""
    import librosa

    audio_array = np.asarray(audio_array)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=0)
    if orig_sr != TARGET_SR:
        audio_array = librosa.resample(
            audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=TARGET_SR
        )
    return audio_array.astype(np.float32)


def is_valid_sample(audio_array: np.ndarray, transcript: str) -> bool:
    duration = len(audio_array) / TARGET_SR
    if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
        return False
    if not transcript or len(transcript.strip()) < 2:
        return False
    return True


def unified(audio: np.ndarray, sentence: str, source: str) -> dict[str, Any]:
    return {
        "audio": {"array": audio, "sampling_rate": TARGET_SR},
        "sentence": sentence,
        "source": source,
    }
