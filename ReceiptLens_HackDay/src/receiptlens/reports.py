"""Persistence helpers, avoid including receipt photos in public reports."""
from pathlib import Path
import json
from .metrics import compare_results


def save_report(plain_df, tuned_df, config, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plain_df.to_csv(out / "baseline_predictions.csv", index=False)
    tuned_df.to_csv(out / "finetuned_predictions.csv", index=False)
    comparison = compare_results(plain_df, tuned_df)
    comparison.to_csv(out / "comparison.csv")
    config.save(out / "experiment_config.json")
    return comparison


def plot_comparison(comparison, out_path="outputs/accuracy_comparison.png"):
    import matplotlib.pyplot as plt
    axes = comparison[["Plain (%)", "Fine-tuned (%)"]].plot(kind="bar", figsize=(11, 5))
    axes.set_ylim(0, 100)
    axes.set_ylabel("Accuracy (%)")
    axes.set_title("Gemma 4 E2B: Plain vs Fine-tuned — SROIE")
    axes.tick_params(axis="x", labelrotation=30)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()
    return out_path
