"""LoRA fine-tune Whisper on the merged Bengali/BD-English corpus.

Debug (default): whisper-small, fp16, 100 samples, 5 steps, r=64 — Colab T4 (15 GB).
Full (--full):   whisper-large-v3, bf16, whole dataset, 10 epochs, r=128 — DGX Spark.

The profile picks the checkpoint and the precision as well as the trainer
settings, so the pipeline is rehearsed on a small model and only the profile
changes for the real run. Both live in configs/*.yaml.

Note: large-v3 uses 128 mel bins where small/medium use 80, so features are
extracted per-run from the processor of whatever checkpoint is loaded — never
reuse a feature cache across profiles.

Usage:
    python scripts/05_train.py                                  # T4 smoke test
    python scripts/05_train.py --model openai/whisper-medium     # closer rehearsal
    python scripts/05_train.py --full                            # DGX full run
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
from datasets import Audio, DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from common import REPO_ROOT, TARGET_SR, load_configs


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad log-mel features and labels separately; mask pad tokens with -100."""

    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # The BOS token is prepended by the model; drop it if the tokenizer added one.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_processor(cfg: dict) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(
        cfg["model"]["base_model_id"],
        language=cfg["model"]["language"],
        task=cfg["model"]["task"],
    )


def load_model(cfg: dict, lora_cfg: dict) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["model"]["base_model_id"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map="auto",
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.generation_config.forced_decoder_ids = None
    # Needed alongside gradient_checkpointing, otherwise no grads reach the LoRA layers.
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=getattr(TaskType, lora_cfg["task_type"]),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def prepare_dataset(dsd: DatasetDict, processor: WhisperProcessor, train_cfg: dict) -> DatasetDict:
    dsd = dsd.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    def encode(batch: dict) -> dict:
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    if train_cfg["max_train_samples"]:
        dsd["train"] = dsd["train"].select(
            range(min(train_cfg["max_train_samples"], len(dsd["train"])))
        )
    if train_cfg["max_eval_samples"]:
        dsd["validation"] = dsd["validation"].select(
            range(min(train_cfg["max_eval_samples"], len(dsd["validation"])))
        )

    return dsd.map(encode, remove_columns=dsd["train"].column_names, num_proc=1)


def build_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


def build_training_args(cfg: dict) -> Seq2SeqTrainingArguments:
    t = cfg["train"]
    return Seq2SeqTrainingArguments(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        max_steps=t["max_steps"],
        num_train_epochs=t["num_train_epochs"],
        eval_strategy="steps",          # renamed from evaluation_strategy in transformers>=4.46
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_strategy="steps",
        save_total_limit=t["save_total_limit"],
        logging_steps=t["logging_steps"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        predict_with_generate=t["predict_with_generate"],
        generation_max_length=t["generation_max_length"],
        gradient_checkpointing=t["gradient_checkpointing"],
        # Non-reentrant checkpointing — the reentrant path silently drops LoRA grads.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[t["report_to"]],
        seed=t["seed"],
        remove_unused_columns=False,    # PEFT-wrapped models need the extra columns kept
        label_names=["labels"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="DGX profile (default: debug)")
    parser.add_argument("--model", default=None, help="override the profile's checkpoint")
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()

    cfg, lora_cfg = load_configs(debug=not args.full, model_override=args.model)
    print(
        f"profile={cfg['profile']}  model={cfg['model']['base_model_id']}  "
        f"dtype={cfg['model']['torch_dtype']}  "
        f"bf16={cfg['train']['bf16']} fp16={cfg['train']['fp16']}  "
        f"lora r={lora_cfg['r']} alpha={lora_cfg['lora_alpha']}"
    )

    dataset_dir = Path(args.dataset_dir or REPO_ROOT / cfg["data"]["dataset_dir"])
    dsd = load_from_disk(str(dataset_dir))

    processor = load_processor(cfg)
    model = load_model(cfg, lora_cfg)
    dsd = prepare_dataset(dsd, processor, cfg["train"])

    trainer = Seq2SeqTrainer(
        args=build_training_args(cfg),
        model=model,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg["train"]["early_stopping_patience"])
        ],
    )

    trainer.train()
    out = cfg["train"]["output_dir"]
    model.save_pretrained(out)          # LoRA adapter only
    processor.save_pretrained(out)
    print(f"adapter saved → {out}")


if __name__ == "__main__":
    main()
