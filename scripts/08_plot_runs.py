"""Plot training curves for one or more runs, side by side.

Reads log_history out of each run's newest checkpoint-*/trainer_state.json —
Trainer writes the full history into every checkpoint, so the highest-numbered
one carries everything, even after save_total_limit pruned the earlier ones.

Draws three panels (train loss / eval loss / eval WER) rather than overlaying
them on one pair of axes: they're different scales, and a dual-axis chart makes
the crossover point an artifact of the scaling rather than the data.

Usage:
    python scripts/08_plot_runs.py
    python scripts/08_plot_runs.py --runs outputs/whisper-bangla-lora outputs/whisper-bangla-lora-proj
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — must follow the Agg backend selection

from common import REPO_ROOT  # noqa: E402

# Categorical slots 1-2 in fixed order, never cycled (validated pair:
# CVD deltaE 24.7, normal-vision 33.6, both clear of the floors).
SERIES_COLORS = ("#2a78d6", "#eb6834")
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

PANELS = (
    ("loss", "Training loss"),
    ("eval_loss", "Validation loss"),
    ("eval_wer", "Validation WER"),
)


def load_history(run_dir: Path) -> list[dict]:
    states = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not states:
        raise SystemExit(
            f"no checkpoint-*/trainer_state.json under {run_dir} — "
            "the run either never reached its first save_steps or the dir is wrong"
        )
    return json.loads(states[-1].read_text(encoding="utf-8"))["log_history"]


def series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(entry["step"], entry[key]) for entry in history if key in entry]
    return [p[0] for p in points], [p[1] for p in points]


def plot(runs: dict[str, list[dict]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)

    for ax, (key, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        is_eval = key.startswith("eval")

        for index, (name, history) in enumerate(runs.items()):
            steps, values = series(history, key)
            if not steps:
                continue
            ax.plot(
                steps,
                values,
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                linewidth=2,
                label=name,
                # Eval points are sparse enough to mark individually; the
                # per-logging-step train loss would turn into a solid smear.
                marker="o" if is_eval else None,
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
            if key == "eval_wer":
                best_step, best_wer = min(zip(steps, values), key=lambda p: p[1])
                ax.annotate(
                    f"{best_wer:.3f}",
                    xy=(best_step, best_wer),
                    xytext=(0, -16),
                    textcoords="offset points",
                    ha="center",
                    color=INK_PRIMARY,
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("step", color=INK_MUTED, fontsize=9)
        ax.grid(True, color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.008, 1.0),
        ncol=len(labels),
        frameon=False,
        labelcolor=INK_PRIMARY,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"curves → {out_path}")


def print_table(runs: dict[str, list[dict]]) -> None:
    """Same numbers as the figure, for reading in a terminal."""
    for name, history in runs.items():
        steps, wers = series(history, "eval_wer")
        _, losses = series(history, "eval_loss")
        print(f"\n{name}")
        print(f"  {'step':>6}  {'eval_loss':>10}  {'eval_wer':>9}")
        for step, loss, wer in zip(steps, losses, wers):
            print(f"  {step:>6}  {loss:>10.4f}  {wer:>9.4f}")
        if wers:
            best_step, best_wer = min(zip(steps, wers), key=lambda p: p[1])
            print(f"  best: {best_wer:.4f} @ step {best_step}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["outputs/whisper-bangla-lora", "outputs/whisper-bangla-lora-proj"],
    )
    parser.add_argument("--out", default="outputs/metrics/training-curves.png")
    args = parser.parse_args()

    runs = {}
    for run in args.runs:
        path = Path(run)
        if not path.is_absolute():
            path = REPO_ROOT / path
        runs[path.name] = load_history(path)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    plot(runs, out)
    print_table(runs)


if __name__ == "__main__":
    main()
