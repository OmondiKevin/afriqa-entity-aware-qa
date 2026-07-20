#!/usr/bin/env python3
"""Generate publication-quality figures for the AfriQA entity-aware QA paper.

Produces:
  1. Bar chart: all configs × 3 metrics (overall)
  2. Grouped bar chart: per-language performance for all models
  3. Delta decomposition chart (exposure vs task) for mT5 and ByT5
  4. Per-language heatmap
  5. LoRA vs full fine-tuning comparison

Saves to outputs/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

METRICS_DIR = Path("outputs_colab/metrics")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Config order and display labels
CONFIGS = [
    ("baseline_mt5_test_metrics", "Baseline\nmT5 (1x)"),
    ("matchedqa_mt5_test_metrics", "Matched-Vol\nmT5 (20x)"),
    ("multitask_mt5_test_qa_only_metrics", "Multitask\nmT5"),
    ("multitask_mt5_lora_test_qa_only_metrics", "LoRA\nmT5"),
    ("matchedqa_byt5_test_metrics", "Matched-Vol\nByT5 (20x)"),
    ("multitask_byt5_test_qa_only_metrics", "Multitask\nByT5"),
    ("multitask_byt5_lora_test_qa_only_metrics", "LoRA\nByT5"),
]

# For short labels in tight plots
CONFIGS_SHORT = [
    ("baseline_mt5_test_metrics", "Base mT5"),
    ("matchedqa_mt5_test_metrics", "MV mT5"),
    ("multitask_mt5_test_qa_only_metrics", "MT mT5"),
    ("multitask_mt5_lora_test_qa_only_metrics", "LoRA mT5"),
    ("matchedqa_byt5_test_metrics", "MV ByT5"),
    ("multitask_byt5_test_qa_only_metrics", "MT ByT5"),
    ("multitask_byt5_lora_test_qa_only_metrics", "LoRA ByT5"),
]

LANGUAGES = ["swa", "hau", "yor"]
LANG_NAMES = {"swa": "Swahili", "hau": "Hausa", "yor": "Yoruba"}
METRICS = ["em", "f1", "semantic"]
METRIC_NAMES = {"em": "Exact Match", "f1": "Token F1", "semantic": "Semantic Sim."}

# Color scheme
MT5_COLOR = "#2563EB"  # blue
BYT5_COLOR = "#059669"  # green
LORA_MT5_COLOR = "#93C5FD"  # light blue
LORA_BYT5_COLOR = "#6EE7B7"  # light green
BASELINE_COLOR = "#9CA3AF"  # gray
EXPOSURE_COLOR = "#F59E0B"  # amber
TASK_COLOR = "#8B5CF6"  # purple


def load_all_metrics() -> dict:
    data = {}
    for stem, _ in CONFIGS:
        path = METRICS_DIR / f"{stem}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data[stem] = json.load(f)
    # Also try translation pipeline
    tp = METRICS_DIR / "translation_pipeline_test_metrics.json"
    if tp.exists():
        with open(tp, encoding="utf-8") as f:
            data["translation_pipeline_test_metrics"] = json.load(f)
    return data


def get_colors(stems: list[str]) -> list[str]:
    """Assign colors based on config type."""
    colors = []
    for stem in stems:
        if "baseline" in stem:
            colors.append(BASELINE_COLOR)
        elif "lora" in stem and "byt5" in stem:
            colors.append(LORA_BYT5_COLOR)
        elif "lora" in stem:
            colors.append(LORA_MT5_COLOR)
        elif "byt5" in stem:
            colors.append(BYT5_COLOR)
        else:
            colors.append(MT5_COLOR)
    return colors


def fig1_overall_bar_chart(data: dict) -> None:
    """Bar chart: all configs × 3 metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    available = [(s, l) for s, l in CONFIGS if s in data]
    stems = [s for s, _ in available]
    labels = [l for _, l in available]
    colors = get_colors(stems)

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        values = [data[s]["overall"].get(metric, 0) for s in stems]
        bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)

        # Highlight best
        best_idx = int(np.argmax(values))
        bars[best_idx].set_edgecolor("#000000")
        bars[best_idx].set_linewidth(2)

        ax.set_title(METRIC_NAMES[metric], fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=0, ha="center")
        ax.set_ylim(0, max(values) * 1.15)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Overall QA Performance Across All Configurations (n=848)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIGURES_DIR / "fig1_overall_comparison.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig2_per_language_grouped(data: dict) -> None:
    """Grouped bar chart: per-language performance for top models."""
    # Focus on 4 key configs for readability
    key_configs = [
        ("baseline_mt5_test_metrics", "Baseline mT5"),
        ("matchedqa_mt5_test_metrics", "Matched mT5"),
        ("multitask_mt5_test_qa_only_metrics", "Multitask mT5"),
        ("matchedqa_byt5_test_metrics", "Matched ByT5"),
        ("multitask_byt5_test_qa_only_metrics", "Multitask ByT5"),
    ]
    available = [(s, l) for s, l in key_configs if s in data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bar_width = 0.15
    x = np.arange(len(LANGUAGES))

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        stems_avail = [s for s, _ in available]
        colors = get_colors(stems_avail)

        for i, (stem, label) in enumerate(available):
            per_lang = data[stem].get("per_lang", {})
            values = [per_lang.get(lang, {}).get(metric, 0) for lang in LANGUAGES]
            offset = (i - len(available) / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=label, color=colors[i],
                   edgecolor="white", linewidth=0.5)

        ax.set_title(METRIC_NAMES[metric], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Per-Language QA Performance", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIGURES_DIR / "fig2_per_language_comparison.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig3_delta_decomposition(data: dict) -> None:
    """Delta decomposition chart: exposure vs task for mT5 and ByT5."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = [
        ("mT5", "baseline_mt5_test_metrics", "matchedqa_mt5_test_metrics", "multitask_mt5_test_qa_only_metrics"),
        ("ByT5", "baseline_byt5_test_metrics", "matchedqa_byt5_test_metrics", "multitask_byt5_test_qa_only_metrics"),
    ]

    for ax_idx, (arch, baseline_key, matched_key, multitask_key) in enumerate(configs):
        ax = axes[ax_idx]

        if baseline_key not in data or matched_key not in data or multitask_key not in data:
            ax.set_title(f"{arch} (data missing)", fontsize=12)
            ax.text(0.5, 0.5, "Missing baseline data", ha="center", va="center", transform=ax.transAxes)
            continue

        x = np.arange(len(METRICS))
        bar_width = 0.35

        exposure_deltas = []
        task_deltas = []
        for metric in METRICS:
            b = data[baseline_key]["overall"].get(metric, 0)
            m = data[matched_key]["overall"].get(metric, 0)
            mt = data[multitask_key]["overall"].get(metric, 0)
            exposure_deltas.append(m - b)
            task_deltas.append(mt - m)

        bars1 = ax.bar(x - bar_width / 2, exposure_deltas, bar_width,
                       label="Δ Exposure (Volume)", color=EXPOSURE_COLOR, edgecolor="white")
        bars2 = ax.bar(x + bar_width / 2, task_deltas, bar_width,
                       label="Δ Task (NER Impact)", color=TASK_COLOR, edgecolor="white")

        # Value labels
        for bar, val in zip(bars1, exposure_deltas):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.003 if y >= 0 else y - 0.012,
                    f"{val:+.3f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=8)
        for bar, val in zip(bars2, task_deltas):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.003 if y >= 0 else y - 0.012,
                    f"{val:+.3f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=8)

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
        ax.set_title(f"{arch} Architecture", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_NAMES[m] for m in METRICS], fontsize=10)
        ax.set_ylabel("Delta", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Deconstructing Gains: Exposure Volume vs Entity-Aware Supervision",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIGURES_DIR / "fig3_delta_decomposition.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig4_heatmap(data: dict) -> None:
    """Per-language heatmap across all configs for EM."""
    available = [(s, l) for s, l in CONFIGS_SHORT if s in data]
    stems = [s for s, _ in available]
    labels = [l for _, l in available]

    matrix = []
    for stem in stems:
        row = [data[stem].get("per_lang", {}).get(lang, {}).get("em", 0) for lang in LANGUAGES]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES], fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(LANGUAGES)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Exact Match", fontsize=11)
    ax.set_title("Exact Match by Configuration × Language", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "fig4_em_heatmap.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig5_lora_comparison(data: dict) -> None:
    """LoRA vs full fine-tuning side-by-side."""
    pairs = [
        ("Full mT5", "multitask_mt5_test_qa_only_metrics",
         "LoRA mT5", "multitask_mt5_lora_test_qa_only_metrics"),
        ("Full ByT5", "multitask_byt5_test_qa_only_metrics",
         "LoRA ByT5", "multitask_byt5_lora_test_qa_only_metrics"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        x_pos = 0
        tick_positions = []
        tick_labels_list = []

        for full_label, full_key, lora_label, lora_key in pairs:
            if full_key in data and lora_key in data:
                full_val = data[full_key]["overall"].get(metric, 0)
                lora_val = data[lora_key]["overall"].get(metric, 0)

                ax.bar(x_pos, full_val, 0.4, color=MT5_COLOR if "mt5" in full_key else BYT5_COLOR,
                       edgecolor="white")
                ax.bar(x_pos + 0.5, lora_val, 0.4,
                       color=LORA_MT5_COLOR if "mt5" in lora_key else LORA_BYT5_COLOR,
                       edgecolor="white")

                ax.text(x_pos, full_val + 0.005, f"{full_val:.3f}", ha="center", fontsize=8)
                ax.text(x_pos + 0.5, lora_val + 0.005, f"{lora_val:.3f}", ha="center", fontsize=8)

                tick_positions.extend([x_pos, x_pos + 0.5])
                tick_labels_list.extend([full_label, lora_label])
                x_pos += 1.5

        ax.set_title(METRIC_NAMES[metric], fontsize=13, fontweight="bold")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, fontsize=8, rotation=15, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Full Fine-Tuning vs LoRA (rank-16) Under Multitask Regime",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIGURES_DIR / "fig5_lora_comparison.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    print("Loading metrics...")
    data = load_all_metrics()
    print(f"Found {len(data)} config results\n")

    if not data:
        print("ERROR: No metrics files found in", METRICS_DIR)
        return

    fig1_overall_bar_chart(data)
    fig2_per_language_grouped(data)
    fig3_delta_decomposition(data)
    fig4_heatmap(data)
    fig5_lora_comparison(data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
