"""Merge the LoRA adapter into the base model and convert to CTranslate2.

Result loads directly in faster-whisper for production serving.
Note: the merge runs in float32 on CPU (safest); quantization happens in the
CTranslate2 conversion step, not the merge.

The base checkpoint must match the adapter's, so pass --full when exporting a
large-v3 run (the default resolves to the debug profile's whisper-small).

Usage:
    python scripts/07_export.py --adapter outputs/whisper-bangla-lora-debug
    python scripts/07_export.py --full --adapter outputs/whisper-bangla-lora
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import REPO_ROOT, load_configs


def merge_adapter(base_model_id: str, adapter: str, merged_dir: Path) -> None:
    base = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.float32
    )
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    WhisperProcessor.from_pretrained(base_model_id).save_pretrained(str(merged_dir))
    print(f"merged model → {merged_dir}")


def convert_ct2(merged_dir: Path, output_dir: Path, quantization: str) -> None:
    converter = shutil.which("ct2-transformers-converter")
    if converter is None:
        raise SystemExit("ct2-transformers-converter not found — pip install ctranslate2")
    subprocess.run(
        [
            converter,
            "--model", str(merged_dir),
            "--output_dir", str(output_dir),
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
            "--quantization", quantization,
            "--force",
        ],
        check=True,
    )
    print(f"CTranslate2 model → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/whisper-bangla-lora")
    parser.add_argument("--merged-dir", default="outputs/merged-whisper-bangla")
    parser.add_argument("--output-dir", default="outputs/faster-whisper-bangla")
    parser.add_argument("--quantization", default="float16")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--full", action="store_true", help="export the large-v3 profile")
    parser.add_argument("--model", default=None, help="override the profile's checkpoint")
    args = parser.parse_args()

    cfg, _ = load_configs(debug=not args.full, model_override=args.model)
    print(f"merging {args.adapter} into {cfg['model']['base_model_id']}")
    merged_dir = REPO_ROOT / args.merged_dir
    output_dir = REPO_ROOT / args.output_dir

    if not args.skip_merge:
        merge_adapter(cfg["model"]["base_model_id"], args.adapter, merged_dir)
    convert_ct2(merged_dir, output_dir, args.quantization)

    print(
        "\nsmoke test:\n"
        "  from faster_whisper import WhisperModel\n"
        f'  m = WhisperModel("{output_dir}", device="cuda", compute_type="float16")\n'
        '  segments, info = m.transcribe("sample.wav", language="bn")\n'
    )


if __name__ == "__main__":
    main()
