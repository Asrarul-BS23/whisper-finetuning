"""Evaluate the fine-tuned adapter on the test split.

Reports overall WER plus WER on the code-switched slice (samples whose transcript
mixes Bengali and Latin script), since that is the failure mode this run targets.

The base checkpoint comes from the profile (debug = whisper-small, full =
large-v3) and must match the one the adapter was trained on — a LoRA adapter is
not portable between Whisper sizes.

Usage:
    python scripts/06_eval.py --adapter outputs/whisper-bangla-lora-debug --limit 10
    python scripts/06_eval.py --full --adapter outputs/whisper-bangla-lora
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Audio, load_from_disk
from jiwer import wer
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import REPO_ROOT, TARGET_SR, load_configs

BENGALI = re.compile(r"[\u0980-\u09FF]")
LATIN = re.compile(r"[A-Za-z]")


def is_code_switched(text: str) -> bool:
    return bool(BENGALI.search(text)) and bool(LATIN.search(text))


def load_eval_model(base_model_id: str, adapter: str | None, dtype: torch.dtype):
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=dtype, device_map="auto"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def evaluate_model(model, processor, test_dataset, language: str, task: str) -> dict[str, float]:
    predictions, references, cs_flags = [], [], []
    for sample in test_dataset:
        features = processor(
            sample["audio"]["array"], sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            predicted_ids = model.generate(features, language=language, task=task)
        predictions.append(
            processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        )
        references.append(sample["sentence"])
        cs_flags.append(is_code_switched(sample["sentence"]))

    results = {"wer": wer(references, predictions), "n": len(references)}
    cs_ref = [r for r, f in zip(references, cs_flags) if f]
    cs_pred = [p for p, f in zip(predictions, cs_flags) if f]
    if cs_ref:
        results["wer_code_switched"] = wer(cs_ref, cs_pred)
        results["n_code_switched"] = len(cs_ref)

    print(f"Overall WER: {results['wer']:.4f}  (n={results['n']})")
    if "wer_code_switched" in results:
        print(
            f"Code-switched WER: {results['wer_code_switched']:.4f} "
            f"(n={results['n_code_switched']})"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="omit to score the untuned baseline")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--model", default=None, help="override the profile's checkpoint (must match --adapter)"
    )
    args = parser.parse_args()

    cfg, _ = load_configs(debug=not args.full, model_override=args.model)
    print(f"scoring {cfg['model']['base_model_id']} adapter={args.adapter or 'none (baseline)'}")
    dataset_dir = Path(args.dataset_dir or REPO_ROOT / cfg["data"]["dataset_dir"])
    ds = load_from_disk(str(dataset_dir))[args.split].cast_column(
        "audio", Audio(sampling_rate=TARGET_SR)
    )
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    processor = WhisperProcessor.from_pretrained(
        cfg["model"]["base_model_id"],
        language=cfg["model"]["language"],
        task=cfg["model"]["task"],
    )
    model = load_eval_model(
        cfg["model"]["base_model_id"],
        args.adapter,
        getattr(torch, cfg["model"]["torch_dtype"]),
    )
    results = evaluate_model(model, processor, ds, cfg["model"]["language"], cfg["model"]["task"])
    results["base_model_id"] = cfg["model"]["base_model_id"]
    results["adapter"] = args.adapter or ""
    results["split"] = args.split

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"metrics → {out}")


if __name__ == "__main__":
    main()
