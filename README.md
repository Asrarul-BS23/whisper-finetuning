# Whisper Large-v3 LoRA — Bengali/English Code-Switched STT

Fine-tunes `openai/whisper-large-v3` with LoRA (PEFT) on Bengali-English
code-switched speech, for transcribing BD-English mixed professional meetings.
Full rationale and target metrics: [initial-plan.md](initial-plan.md).

## Layout

```
configs/     lora_config.yaml, training_config.yaml — debug + full profiles
scripts/     01..07 pipeline, common.py shared helpers
notebooks/   debug_colab.ipynb (T4 smoke test), full_train_dgx.ipynb (full run)
data/        raw/ (HF cache) · processed/<source> · processed/final · augmented/
outputs/     LoRA adapters, merged model, CTranslate2 export
```

## Setup

Dependencies are managed with **pip-tools**: `requirements/*.in` holds the direct
deps, `requirements/*-*.txt` are the compiled, fully-pinned, hash-verified locks.
Edit the `.in`, never the `.txt`.

```powershell
# local authoring / data inspection (CPU, Windows) — no torch
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements/dev-win.txt
```

```bash
# Colab / DGX — the lock is platform-specific, so compile it there once:
pip install pip-tools
pip-compile --generate-hashes --strip-extras   --output-file=requirements/train-linux-cu124.txt requirements/train.in
pip install -r requirements/train-linux-cu124.txt   # then commit the lock
```

The training lock **must be compiled on Linux** — the CUDA `nvidia-*` wheels
that `torch` pulls in do not exist on Windows, so a lock compiled here would be
silently wrong for Colab/DGX. Until it exists, the notebooks fall back to
`pip install -r requirements/train.in` (direct pins only, transitive unpinned).

Copy `.env.example` → `.env` and set `HF_TOKEN` (Common Voice is gated and
`push_to_hub` needs it).

### Data & model versioning (DVC)

```bash
pip install "dvc[gdrive]"        # already in the locks
dvc init                          # git repo is already initialized
dvc remote add -d storage gdrive://<folder-id>    # or s3://, azure://
```

`dvc.yaml` wires the numbered scripts into a pipeline. Stage names are
per-profile, so **always name the stage** — a bare `dvc repro` would try to run
the large-v3 DGX training too:

```bash
dvc repro build                  # data stages only
dvc repro train@debug            # Colab T4 rehearsal
dvc repro evaluate@debug
dvc repro train@full             # DGX
dvc metrics diff                 # WER before/after, from outputs/metrics/*.json
dvc push                         # datasets + adapters to the remote
```

`configs/*.yaml` are registered as DVC params, so changing `data.mix` or the LoRA
rank invalidates exactly the stages downstream of it and nothing else.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1 Explore | `python scripts/01_explore_datasets.py` | column names / sample rates per source |
| 2 Prepare | `python scripts/02_prepare_data.py --source medibeng [--limit 100]` | `data/processed/medibeng` |
| 3 Augment | `python scripts/03_augment.py --source medibeng` | `data/augmented/medibeng` |
| 4 Build | `python scripts/04_build_dataset.py [--push user/repo]` | `data/processed/final` (train/val/test) |
| 5 Train | `python scripts/05_train.py [--full]` | LoRA adapter in `outputs/` |
| 6 Eval | `python scripts/06_eval.py --adapter outputs/whisper-bangla-lora` | overall + code-switched WER, `--out-json` for DVC metrics |
| 7 Export | `python scripts/07_export.py --adapter outputs/whisper-bangla-lora` | `outputs/faster-whisper-bangla` |

**Run Step 1 first.** Steps 2-4 pick the transcript column from
`TRANSCRIPT_KEYS` in `02_prepare_data.py`; the entries are best guesses per
source and must be corrected against real Step 1 output before a full run.

## Debug → full

The profile decides the **checkpoint and the precision**, not just the trainer
settings, so the pipeline is rehearsed cheaply and only the flag changes:

| | debug (default) | `--full` |
|---|---|---|
| Model | `openai/whisper-medium` (769M) | `openai/whisper-large-v3` (1.55B) |
| Precision | fp32 weights + fp16 autocast | bf16 |
| Data | 100 train / 10 eval samples | whole dataset |
| Steps | 5 | 10 epochs, early stop |
| Effective batch | 2 x 8 accum = 16 | 64 x 1 accum = 64 |
| LoRA | r=64, alpha=128 | r=128, alpha=256 |
| Target | Colab T4, 15 GB | DGX Spark |

Why not large-v3 on Colab: a 15 GB T4 is Turing — **no bfloat16 at all**, and too
tight for 1.55B params plus activations. Medium is the closest rehearsal that
fits; drop to `--model openai/whisper-small` if the T4 OOMs or iteration feels
slow. `--model` overrides the checkpoint on `05`/`06`/`07` without touching
anything else.

Gradient accumulation carries the effective batch on both: the T4 cannot hold
more than ~2 30-second samples of medium at once, so 8 accumulation steps reach
an effective 16. The DGX uses 64 x 1 = 64. Note that `max_steps: 5` counts
*optimizer* steps, so the smoke test does 5 x 16 = 80 forward passes.

Two constraints this creates:

- A LoRA adapter is **not portable across Whisper sizes** — pass the same
  profile/`--model` to `06_eval.py` and `07_export.py` that you trained with.
- large-v3 uses **128 mel bins**, small/medium use 80. Features are extracted at
  train time from the loaded checkpoint's processor, so this is handled — but
  never copy a cached feature set from the T4 run to the DGX.

Work through the checklist in `notebooks/debug_colab.ipynb` before the DGX run;
all boxes must pass. What it does *not* prove: large-v3 memory headroom, bf16
numerics, or absolute WER.

## Key decisions

- Language token `bn`, not `en` — decisive for code-switching accuracy.
- LoRA over full fine-tune — a 20-30 hr corpus would overfit a full tune.
- bf16 on the DGX (Ampere/Blackwell native, no grad scaler); fp16 autocast on the
  T4 out of necessity. Debug weights load in **fp32** — loading fp16 weights
  breaks the grad scaler with "cannot unscale FP16 gradients".
- Augment MediBeng only — IndicVoices and Common Voice are already speaker-diverse.
- Export CTranslate2 float16 for faster-whisper serving.

## Deviations from the plan

- `evaluation_strategy` → `eval_strategy` (renamed in transformers ≥ 4.46).
- Export uses `ct2-transformers-converter`, not the `ct2-opus-mt-*` converter
  named in the plan's code block (the plan flags this itself).
- Hyperparameters moved from script constants into `configs/*.yaml` so the
  debug/full switch is one flag rather than three edited lines.
- Added `scripts/common.py` (shared cleaning/resampling/config loading),
  `requirements-dev.txt`, and a code-switched WER slice in Step 6.
- The plan's debug run trained large-v3 on a T4 with `bf16=True`. That cannot
  work (Turing has no bf16, and 15 GB is too tight), so the debug profile trains
  whisper-medium in fp16 and the model became a per-profile setting.
- `gradient_checkpointing_kwargs={"use_reentrant": False}` plus
  `enable_input_require_grads()` — the reentrant checkpointing path silently
  drops LoRA gradients.
