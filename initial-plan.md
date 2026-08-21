Here's the complete self-contained implementation plan:

---

# Whisper Large-v3 Fine-tuning Plan
## Bengali-English Code-Switched STT

---

## Objective
Fine-tune `openai/whisper-large-v3` using LoRA (PEFT) on Bengali-English code-switched speech data to improve transcription accuracy for BD-English mixed professional meetings.

---

## Models

| Role | Model | HuggingFace ID |
|---|---|---|
| Base model | Whisper Large-v3 | `openai/whisper-large-v3` |
| LoRA adapter | Train from scratch | saved locally |
| Inference | faster-whisper | via CTranslate2 export |

---

## Datasets

| Dataset | HF ID | Size | Purpose |
|---|---|---|---|
| MediBeng | `pr0mila-gh0sh/MediBeng` | Small | BD-English code-switched, primary |
| IndicVoices Bengali | `ai4bharat/IndicVoices` | Large | Speaker diversity, BD accent |
| Common Voice Bengali | `mozilla-foundation/common_voice_17_0` (bn) | Medium | Clean Bengali speech |
| FLEURS Bengali | `google/fleurs` (bn_in) | Medium | Additional Bengali coverage |

---

## Project Structure

```
whisper-bangla-finetune/
├── data/
│   ├── raw/                    # downloaded datasets cached here
│   ├── processed/              # cleaned, resampled, chunked
│   └── augmented/              # augmentation applied
│
├── scripts/
│   ├── 01_explore_datasets.py  # explore each dataset structure
│   ├── 02_prepare_data.py      # clean + normalize + chunk
│   ├── 03_augment.py           # audio augmentation
│   ├── 04_build_dataset.py     # merge all sources → HF dataset
│   ├── 05_train.py             # LoRA fine-tuning script
│   ├── 06_eval.py              # WER evaluation
│   └── 07_export.py            # export to CTranslate2 for faster-whisper
│
├── notebooks/
│   ├── debug_colab.ipynb       # Colab debug notebook (100 samples, 5 steps)
│   └── full_train_dgx.ipynb    # DGX Spark full training notebook
│
├── configs/
│   ├── lora_config.yaml        # LoRA hyperparameters
│   └── training_config.yaml    # training hyperparameters
│
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
# requirements.txt
torch==2.6.0
transformers==4.51.0
peft==0.15.0
datasets==3.5.0
accelerate==1.6.0
evaluate==0.4.3
jiwer==3.0.5                    # WER computation
audiomentations==0.38.0         # audio augmentation
soundfile==0.12.1
librosa==0.11.0
ctranslate2==4.5.0              # export to faster-whisper
faster-whisper==1.1.0           # inference
huggingface_hub==0.28.0
```

```bash
pip install -r requirements.txt
```

---

## Step 1 — Explore Datasets (`01_explore_datasets.py`)

```python
"""
Purpose: understand structure of each dataset before processing
Run this first to know column names, audio format, sample rates
"""
from datasets import load_dataset

# MediBeng — Bengali-English code-switched
medibeng = load_dataset("pr0mila-gh0sh/MediBeng", split="train")
print("MediBeng columns:", medibeng.column_names)
print("MediBeng sample:", medibeng[0])
print("MediBeng size:", len(medibeng))

# IndicVoices Bengali subset
indicvoices = load_dataset(
    "ai4bharat/IndicVoices",
    "bn",                        # Bengali subset
    split="train",
    trust_remote_code=True
)
print("IndicVoices columns:", indicvoices.column_names)
print("IndicVoices sample:", indicvoices[0])

# Common Voice Bengali
common_voice = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "bn",
    split="train",
    trust_remote_code=True
)
print("CommonVoice columns:", common_voice.column_names)

# FLEURS Bengali
fleurs = load_dataset(
    "google/fleurs",
    "bn_in",
    split="train",
    trust_remote_code=True
)
print("FLEURS columns:", fleurs.column_names)
```

---

## Step 2 — Data Preparation (`02_prepare_data.py`)

```python
"""
Purpose: normalize all datasets into unified format
Output schema: {"audio": {"array": np.array, "sampling_rate": 16000}, "sentence": str}
Requirements:
  - Audio: 16kHz mono, max 30 seconds per chunk
  - Transcript: cleaned unicode, no noise tags
  - Remove samples: duration < 1s, duration > 30s, empty transcript
"""
from __future__ import annotations
import re
import numpy as np
import librosa
from datasets import load_dataset, Dataset, Audio
from typing import Any

TARGET_SR = 16000
MAX_DURATION_S = 30
MIN_DURATION_S = 1

def clean_transcript(text: str) -> str:
    """Remove noise tags, normalize Bengali unicode, strip extra whitespace."""
    text = re.sub(r'\[.*?\]', '', text)       # remove [noise], [laughter] etc
    text = re.sub(r'\(.*?\)', '', text)       # remove (inaudible) etc
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def resample_audio(audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample to 16kHz mono."""
    if orig_sr != TARGET_SR:
        audio_array = librosa.resample(
            audio_array, orig_sr=orig_sr, target_sr=TARGET_SR
        )
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=0)
    return audio_array.astype(np.float32)

def is_valid_sample(audio_array: np.ndarray, transcript: str) -> bool:
    """Filter out bad samples."""
    duration = len(audio_array) / TARGET_SR
    if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
        return False
    if not transcript or len(transcript.strip()) < 2:
        return False
    return True

def process_medibeng(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize MediBeng sample — check column names from Step 1 first."""
    audio = resample_audio(
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"]
    )
    transcript = clean_transcript(sample["sentence"])  # adjust key if needed
    if not is_valid_sample(audio, transcript):
        return None
    return {
        "audio": {"array": audio, "sampling_rate": TARGET_SR},
        "sentence": transcript,
        "source": "medibeng"
    }

def process_indicvoices(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize IndicVoices sample — adjust column keys after Step 1."""
    audio = resample_audio(
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"]
    )
    transcript = clean_transcript(sample["transcript"])  # adjust key if needed
    if not is_valid_sample(audio, transcript):
        return None
    return {
        "audio": {"array": audio, "sampling_rate": TARGET_SR},
        "sentence": transcript,
        "source": "indicvoices"
    }

def process_common_voice(sample: dict[str, Any]) -> dict[str, Any] | None:
    audio = resample_audio(
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"]
    )
    transcript = clean_transcript(sample["sentence"])
    if not is_valid_sample(audio, transcript):
        return None
    return {
        "audio": {"array": audio, "sampling_rate": TARGET_SR},
        "sentence": transcript,
        "source": "common_voice"
    }
```

---

## Step 3 — Augmentation (`03_augment.py`)

```python
"""
Purpose: synthetically expand speaker diversity
Apply to MediBeng and real meeting recordings only
Do NOT augment IndicVoices/CommonVoice (already diverse)
"""
from __future__ import annotations
import numpy as np
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise

AUGMENT_PIPELINE = Compose([
    PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
])

def augment_sample(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    n_augments: int = 4
) -> list[np.ndarray]:
    """Generate N augmented versions of one audio sample."""
    return [
        AUGMENT_PIPELINE(samples=audio_array, sample_rate=sample_rate)
        for _ in range(n_augments)
    ]
```

---

## Step 4 — Build Final Dataset (`04_build_dataset.py`)

```python
"""
Purpose: merge all sources, split train/val/test, push to HF
Final dataset composition target:
  - MediBeng (+ augmented): ~20%
  - IndicVoices Bengali:    ~40%
  - Common Voice Bengali:   ~25%
  - FLEURS Bengali:         ~15%
"""
from datasets import concatenate_datasets, DatasetDict

def build_final_dataset(
    medibeng_processed: list,
    indicvoices_processed: list,
    common_voice_processed: list,
    fleurs_processed: list,
) -> DatasetDict:
    combined = concatenate_datasets([
        medibeng_processed,
        indicvoices_processed,
        common_voice_processed,
        fleurs_processed,
    ])
    combined = combined.shuffle(seed=42)
    split = combined.train_test_split(test_size=0.1, seed=42)
    val_test = split["test"].train_test_split(test_size=0.5, seed=42)
    return DatasetDict({
        "train": split["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

# Push to HuggingFace private repo
# dataset.push_to_hub("your-username/bangla-en-meeting-asr", private=True)
```

---

## Step 5 — LoRA Fine-tuning (`05_train.py`)

```python
"""
Purpose: fine-tune Whisper large-v3 with LoRA
Debug mode: max_steps=5, dataset subset 100 samples
Full mode:  num_train_epochs=10, full dataset
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# ── Config ────────────────────────────────────────────────────
MODEL_ID = "openai/whisper-large-v3"
LANGUAGE = "bn"                  # Bengali prompt token — critical for code-switching
TASK = "transcribe"
DEBUG_MODE = True                # set False for DGX full run

LORA_CONFIG = LoraConfig(
    r=64 if DEBUG_MODE else 128,        # r=128 on DGX Spark
    lora_alpha=128 if DEBUG_MODE else 256,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

TRAINING_ARGS = Seq2SeqTrainingArguments(
    output_dir="./whisper-bangla-lora",
    per_device_train_batch_size=2 if DEBUG_MODE else 32,
    per_device_eval_batch_size=2 if DEBUG_MODE else 16,
    gradient_accumulation_steps=8 if DEBUG_MODE else 2,
    learning_rate=1e-4,
    warmup_steps=50,
    max_steps=5 if DEBUG_MODE else -1,          # 5 steps debug, full epochs otherwise
    num_train_epochs=1 if DEBUG_MODE else 10,
    evaluation_strategy="steps",
    eval_steps=50 if not DEBUG_MODE else 5,
    save_steps=200 if not DEBUG_MODE else 5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    fp16=False,
    bf16=True,                              # bfloat16 — Blackwell native on DGX
    predict_with_generate=True,
    generation_max_length=225,
    report_to="mlflow",                     # MLflow tracking
    gradient_checkpointing=True,            # saves VRAM on Colab T4
)

# ── Model + LoRA ──────────────────────────────────────────────
def load_model() -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return model

# ── WER Metric ────────────────────────────────────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

---

## Step 6 — Eval (`06_eval.py`)

```python
"""
Purpose: evaluate fine-tuned model on test split
Report: WER overall + WER on code-switched segments specifically
"""
from jiwer import wer

def evaluate_model(model, processor, test_dataset) -> dict[str, float]:
    predictions, references = [], []
    for sample in test_dataset:
        # run inference
        input_features = processor(
            sample["audio"]["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        pred_text = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        predictions.append(pred_text)
        references.append(sample["sentence"])

    overall_wer = wer(references, predictions)
    print(f"Overall WER: {overall_wer:.4f}")
    return {"wer": overall_wer}
```

---

## Step 7 — Export to faster-whisper (`07_export.py`)

```python
"""
Purpose: convert fine-tuned LoRA model to CTranslate2 format
Output: compatible with faster-whisper for production serving
"""
import subprocess
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

def export_to_faster_whisper(
    base_model_id: str,
    lora_checkpoint: str,
    output_dir: str,
) -> None:
    # Step 1: merge LoRA into base model
    base = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base, lora_checkpoint)
    merged = model.merge_and_unload()
    merged.save_pretrained("./merged-whisper-bangla")

    # Step 2: convert to CTranslate2
    subprocess.run([
        "ct2-opus-mt-en-de-converter",     # wrong — use correct converter below
        "--model", "./merged-whisper-bangla",
        "--output_dir", output_dir,
        "--quantization", "float16",
    ])

# Correct CTranslate2 conversion command:
# ct2-transformers-converter \
#   --model ./merged-whisper-bangla \
#   --output_dir ./faster-whisper-bangla \
#   --force \
#   --quantization float16
```

---

## Debug Checklist (Colab T4 — run before DGX)

```
□ Step 1: All 4 datasets load without error
□ Step 2: process_* functions return correct schema
□ Step 3: Augmented audio sounds reasonable (listen to 2-3 samples)
□ Step 4: Final dataset has train/val/test splits, correct sizes
□ Step 5: Training starts, loss appears, no NaN/inf, no OOM on 100 samples
□ Step 5: 5 debug steps complete, model saves checkpoint
□ Step 6: WER computable on 10 test samples
□ Step 7: Export runs, faster-whisper loads exported model
All 8 pass → go to DGX full run
```

---

## DGX Spark Full Run Changes

```python
# Only 3 things change from debug to full:
DEBUG_MODE = False               # line 1
LORA_CONFIG.r = 128              # line 2 — higher rank, more capacity
LORA_CONFIG.lora_alpha = 256     # line 3
# Everything else stays identical
```

---

## Expected Results

| Metric | Baseline (large-v3) | After fine-tuning |
|---|---|---|
| WER Bengali clean | ~35% | ~15-20% |
| WER BD-English mixed | ~45% | ~20-28% |
| Domain terms accuracy | Poor | Good |
| Training time Colab T4 | — | ~8-12 hrs full |
| Training time DGX Spark | — | ~1.5-2 hrs full |

---

## Key Decisions to Remember

| Decision | Value | Reason |
|---|---|---|
| Language token | `bn` not `en` | Critical for code-switching accuracy |
| LoRA not full fine-tune | 20-30hr dataset too small | Prevents overfitting |
| LoRA rank | 64 (Colab), 128 (DGX) | Memory vs capacity tradeoff |
| bfloat16 | Always | Blackwell/Ampere native, stable |
| Early stopping patience | 3 evals | Stops overfitting automatically |
| Augmentation | MediBeng only | IndicVoices already diverse |
| Export format | CTranslate2 float16 | faster-whisper compatible |
---

# Revisions Log

Decisions made during setup that supersede the plan above. The plan's intent is
unchanged; these are corrections found while making it runnable.

## Batch size & gradient accumulation

| | Colab T4 (debug) | DGX Spark (full) |
|---|---|---|
| Model | `openai/whisper-medium` (769M) | `openai/whisper-large-v3` (1.55B) |
| `per_device_train_batch_size` | 2 | 64 |
| `gradient_accumulation_steps` | 8 | 1 |
| **Effective batch** | 16 | 64 |

**T4 — accumulation is load-bearing.** At 30-second inputs the card holds ~2
samples of medium at once, so 8 accumulation steps are the only way to reach a
usable effective batch. If it OOMs, drop to `whisper-small`; do **not** cut the
accumulation steps, or the smoke test stops exercising the path the DGX uses.

**DGX — accumulation is not needed.** 128 GB of unified memory holds a real
batch, so `gradient_accumulation_steps: 1` with `per_device_train_batch_size: 64`
hits the same effective 64 without the extra optimizer bookkeeping. Accumulation
exists to fake a big batch on a small card; there is no small card here.

Caveats to check on the first DGX run:
- DGX Spark is memory-rich but **bandwidth-bound** (~273 GB/s unified LPDDR5X,
  not HBM). Bigger batches help throughput only until bandwidth saturates —
  measure samples/sec at 32 / 64 / 96 before settling on 64.
- Consider `gradient_checkpointing: false` on the DGX. It was there to save T4
  VRAM and costs ~30% throughput; with 128 GB there is likely no need to pay it.
- `max_steps` counts **optimizer** steps, not forward passes. With accumulation
  at 8, the debug profile's `max_steps: 5` is 80 forward passes.
- `learning_rate: 1e-4` is currently shared across both profiles despite a 4x
  effective-batch gap. Defensible for LoRA (far less batch-sensitive than a full
  fine-tune) but it is an untested assumption, not a validated choice. If the
  DGX batch is raised past 64, revisit — the usual rule is to scale LR with
  sqrt(batch ratio).

## Precision — the plan's "bfloat16 always" cannot hold on Colab

A 15 GB Colab GPU is a **T4 (Turing): no bfloat16 unit at all**, and too tight
for 1.55B params plus activations. So precision became per-profile:

- **T4:** fp32 weights + fp16 autocast. Loading fp16 *weights* breaks the grad
  scaler ("Attempting to unscale FP16 gradients") — the weights must stay fp32.
- **DGX:** bf16 as planned (Blackwell native, no grad scaler needed).

## Other corrections

- `evaluation_strategy` → `eval_strategy` (renamed in transformers >= 4.46).
- Export uses `ct2-transformers-converter`, not `ct2-opus-mt-en-de-converter`.
- `gradient_checkpointing_kwargs={"use_reentrant": False}` plus
  `enable_input_require_grads()` — the reentrant checkpointing path silently
  drops LoRA gradients, giving a loss that never moves rather than an error.
- `remove_unused_columns=False` — required for PEFT-wrapped models.
- **A LoRA adapter is not portable across Whisper sizes.** Eval and export must
  load the same base checkpoint that was trained.
- **large-v3 uses 128 mel bins; small/medium use 80.** Features are extracted at
  train time from the loaded checkpoint's processor, so this is handled — but a
  cached feature set must never be copied from the T4 run to the DGX.
- Hyperparameters moved out of script constants into `configs/*.yaml`, so
  debug → full is one flag instead of three hand-edited lines.

## Tooling added after the plan

**pip-tools** — `requirements/*.in` (direct deps) compile to hash-pinned locks.
This immediately caught a conflict the plan would have hit on the first Colab
cell: `audiomentations==0.38.0` requires `librosa<0.11.0`, but the plan pinned
`librosa==0.11.0`, so `pip install -r requirements.txt` was unsatisfiable.
Resolved by moving to `audiomentations==0.43.1`. The training lock must be
compiled **on Linux** — `torch`'s `nvidia-*` CUDA wheels do not exist on Windows,
so a Windows-compiled lock is silently wrong for Colab/DGX.

Verify on the first Colab run: the lock resolves to `numpy==2.4.6` /
`numba==0.67.0`. That pairing is self-consistent, but the audio stack on numpy 2
is a common breakage point — if librosa or numba misbehaves, constrain
`numpy<2.3` in `requirements/train.in` and recompile.

**DVC** — `dvc.yaml` wires the numbered scripts into a pipeline, with
`configs/*.yaml` registered as params so a change to `data.mix` or the LoRA rank
invalidates exactly the downstream stages. Fits this project well: the datasets
are large and immutable, the LoRA adapters are small enough to version freely,
and the numbered scripts already form a clean DAG.

Two things it does not solve, worth knowing up front:
- **Upstream datasets are not tracked as deps.** They come from the HF Hub, so a
  `prepare` stage will not re-run when the source changes. Force it manually:
  `dvc repro -f prepare@<source>`.
- **Stage names are per-profile.** A bare `dvc repro` would try to run the
  large-v3 DGX training. Always name the stage: `dvc repro train@debug`.

`03_augment.py` is now seeded from `common.seed` — audiomentations draws from the
global RNGs, so without it every run produced different audio and invalidated the
DVC cache.
