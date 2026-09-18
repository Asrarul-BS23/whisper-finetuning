"""Push 06_eval.py's test-set metrics into MLflow.

Training logs itself to MLflow live (report_to: mlflow), but 06_eval.py only
writes JSON — so the held-out test numbers never reach the UI. This reads
outputs/metrics/*.json and logs them, with no GPU re-run.

Two modes:
  * default — one new MLflow run per JSON file, in their own experiment. Good
    for a side-by-side baseline/run-1/run-2 comparison view.
  * --attach <json-stem>=<mlflow-run-id> — append the metrics to an existing
    training run instead, so that run's page carries both its curves and its
    final test numbers. Grab the run id from the MLflow UI's run page.

Usage:
    python scripts/09_log_eval_to_mlflow.py
    python scripts/09_log_eval_to_mlflow.py --attach full=8a692d03827f4996bd47592d8b21387e
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from common import REPO_ROOT

METRIC_KEYS = ("wer", "cer", "wer_code_switched", "cer_code_switched")
COUNT_KEYS = ("n", "n_code_switched")
PARAM_KEYS = ("base_model_id", "adapter", "split")

METRICS_DIR = REPO_ROOT / "outputs" / "metrics"


def payload_metrics(payload: dict) -> dict[str, float]:
    return {
        f"test_{key}": float(payload[key])
        for key in METRIC_KEYS + COUNT_KEYS
        if key in payload
    }


def payload_params(payload: dict, stem: str) -> dict[str, str]:
    params = {"eval_source": stem}
    for key in PARAM_KEYS:
        if key in payload:
            params[key] = str(payload[key]) or "none (baseline)"
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--experiment", default="whisper-bangla-eval")
    parser.add_argument(
        "--attach",
        nargs="*",
        default=[],
        metavar="STEM=RUN_ID",
        help="append a JSON's metrics to an existing MLflow run instead of making a new one",
    )
    args = parser.parse_args()

    attach = dict(pair.split("=", 1) for pair in args.attach)
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else METRICS_DIR
    files = sorted(metrics_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no *.json under {metrics_dir} — run 06_eval.py with --out-json first")

    mlflow.set_experiment(args.experiment)

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics, params = payload_metrics(payload), payload_params(payload, path.stem)
        if not metrics:
            print(f"skipped {path.name} — no recognized metric keys")
            continue

        run_id = attach.get(path.stem)
        with mlflow.start_run(run_id=run_id, run_name=None if run_id else path.stem):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        target = f"existing run {run_id}" if run_id else f"new run '{path.stem}'"
        print(f"{path.name} → {target}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
