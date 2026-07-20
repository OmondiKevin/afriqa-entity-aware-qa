#!/usr/bin/env python3
"""Analyze per-language deltas across all experiment configurations.

Reads all metrics JSON files from outputs_colab/metrics/ and produces:
  - Overall comparison table (all configs × 3 metrics)
  - Per-language breakdown table
  - Exposure delta vs Task delta decomposition for mT5 and ByT5
  - CSV and JSON outputs for paper tables
"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any


METRICS_DIR = Path("outputs_colab/metrics")
OUTPUT_DIR = Path("outputs/analysis")

# Map config stems to human-readable names and ordering
CONFIG_ORDER = [
    ("baseline_mt5_test_metrics", "Baseline mT5 (1x)"),
    ("baseline_byt5_test_metrics", "Baseline ByT5 (1x)"),
    ("matchedqa_mt5_test_metrics", "Matched-Vol mT5 (20x)"),
    ("matchedqa_byt5_test_metrics", "Matched-Vol ByT5 (20x)"),
    ("multitask_mt5_test_qa_only_metrics", "Multitask mT5 (NER→QA)"),
    ("multitask_byt5_test_qa_only_metrics", "Multitask ByT5 (NER→QA)"),
    ("multitask_mt5_lora_test_qa_only_metrics", "mT5 LoRA Multitask"),
    ("multitask_byt5_lora_test_qa_only_metrics", "ByT5 LoRA Multitask"),
    ("translation_pipeline_test_metrics", "Translation Pipeline"),
]

LANGUAGES = ["swa", "hau", "yor"]
METRICS = ["em", "f1", "semantic"]


def load_all_metrics(metrics_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all available metrics JSON files."""
    result = {}
    for stem, _ in CONFIG_ORDER:
        path = metrics_dir / f"{stem}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                result[stem] = json.load(f)
    return result


def format_metric(val: float | None, digits: int = 3) -> str:
    if val is None:
        return "—"
    return f"{val:.{digits}f}"


def overall_comparison_table(data: dict[str, dict]) -> list[dict]:
    """Build overall comparison across configs."""
    rows = []
    for stem, label in CONFIG_ORDER:
        if stem not in data:
            continue
        overall = data[stem].get("overall", {})
        rows.append({
            "config": label,
            "n": overall.get("n", 0),
            "em": overall.get("em"),
            "f1": overall.get("f1"),
            "semantic": overall.get("semantic"),
        })
    return rows


def per_language_table(data: dict[str, dict]) -> list[dict]:
    """Build per-language breakdown for each config."""
    rows = []
    for stem, label in CONFIG_ORDER:
        if stem not in data:
            continue
        per_lang = data[stem].get("per_lang", {})
        for lang in LANGUAGES:
            stats = per_lang.get(lang, {})
            rows.append({
                "config": label,
                "lang": lang,
                "n": stats.get("n", 0),
                "em": stats.get("em"),
                "f1": stats.get("f1"),
                "semantic": stats.get("semantic"),
            })
    return rows


def compute_deltas(data: dict[str, dict]) -> dict[str, Any]:
    """Compute exposure delta and task delta for mT5 and ByT5."""
    deltas = {}

    for arch, baseline_key, matched_key, multitask_key in [
        ("mT5", "baseline_mt5_test_metrics", "matchedqa_mt5_test_metrics", "multitask_mt5_test_qa_only_metrics"),
        ("ByT5", "baseline_byt5_test_metrics", "matchedqa_byt5_test_metrics", "multitask_byt5_test_qa_only_metrics"),
    ]:
        arch_deltas = {"overall": {}, "per_lang": {}}

        baseline = data.get(baseline_key, {})
        matched = data.get(matched_key, {})
        multitask = data.get(multitask_key, {})

        if not matched:
            deltas[arch] = {"note": f"Missing matched-volume data ({matched_key})"}
            continue

        # Overall deltas
        for metric in METRICS:
            b_val = baseline.get("overall", {}).get(metric)
            m_val = matched.get("overall", {}).get(metric)
            mt_val = multitask.get("overall", {}).get(metric)

            exposure = (m_val - b_val) if (m_val is not None and b_val is not None) else None
            task = (mt_val - m_val) if (mt_val is not None and m_val is not None) else None
            total = (mt_val - b_val) if (mt_val is not None and b_val is not None) else None

            arch_deltas["overall"][metric] = {
                "baseline": b_val,
                "matched": m_val,
                "multitask": mt_val,
                "delta_exposure": exposure,
                "delta_task": task,
                "delta_total": total,
            }

        # Per-language deltas
        for lang in LANGUAGES:
            lang_deltas = {}
            for metric in METRICS:
                b_val = baseline.get("per_lang", {}).get(lang, {}).get(metric)
                m_val = matched.get("per_lang", {}).get(lang, {}).get(metric)
                mt_val = multitask.get("per_lang", {}).get(lang, {}).get(metric)

                exposure = (m_val - b_val) if (m_val is not None and b_val is not None) else None
                task = (mt_val - m_val) if (mt_val is not None and m_val is not None) else None
                total = (mt_val - b_val) if (mt_val is not None and b_val is not None) else None

                lang_deltas[metric] = {
                    "baseline": b_val,
                    "matched": m_val,
                    "multitask": mt_val,
                    "delta_exposure": exposure,
                    "delta_task": task,
                    "delta_total": total,
                }
            arch_deltas["per_lang"][lang] = lang_deltas

        deltas[arch] = arch_deltas

    return deltas


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_markdown_table(rows: list[dict], title: str) -> None:
    if not rows:
        print(f"\n### {title}\n(No data available)\n")
        return
    keys = list(rows[0].keys())
    print(f"\n### {title}\n")
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---:" if k in METRICS else "---" for k in keys) + " |"
    print(header)
    print(sep)
    for row in rows:
        vals = []
        for k in keys:
            v = row[k]
            if isinstance(v, float):
                vals.append(format_metric(v))
            elif v is None:
                vals.append("—")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")
    print()


def print_delta_table(deltas: dict, arch: str) -> None:
    print(f"\n### Delta Decomposition: {arch}\n")
    if "note" in deltas:
        print(f"*{deltas['note']}*\n")
        return

    # Overall
    print(f"#### Overall ({arch})\n")
    print("| Metric | Baseline | Matched-Vol | Multitask | Δ Exposure | Δ Task | Δ Total |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for metric in METRICS:
        d = deltas["overall"].get(metric, {})
        print(f"| **{metric.upper()}** | {format_metric(d.get('baseline'))} | "
              f"{format_metric(d.get('matched'))} | {format_metric(d.get('multitask'))} | "
              f"{format_metric(d.get('delta_exposure'), 4)} | {format_metric(d.get('delta_task'), 4)} | "
              f"{format_metric(d.get('delta_total'), 4)} |")
    print()

    # Per-language
    for lang in LANGUAGES:
        lang_data = deltas.get("per_lang", {}).get(lang, {})
        if not lang_data:
            continue
        print(f"#### {lang.upper()} ({arch})\n")
        print("| Metric | Baseline | Matched-Vol | Multitask | Δ Exposure | Δ Task | Δ Total |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for metric in METRICS:
            d = lang_data.get(metric, {})
            print(f"| **{metric.upper()}** | {format_metric(d.get('baseline'))} | "
                  f"{format_metric(d.get('matched'))} | {format_metric(d.get('multitask'))} | "
                  f"{format_metric(d.get('delta_exposure'), 4)} | {format_metric(d.get('delta_task'), 4)} | "
                  f"{format_metric(d.get('delta_total'), 4)} |")
        print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metrics from:", METRICS_DIR)
    data = load_all_metrics(METRICS_DIR)
    print(f"Found {len(data)} config results\n")

    # 1. Overall comparison
    overall_rows = overall_comparison_table(data)
    print_markdown_table(overall_rows, "Overall Comparison (All Configs)")
    write_csv(overall_rows, OUTPUT_DIR / "overall_comparison.csv")

    # 2. Per-language breakdown
    lang_rows = per_language_table(data)
    print_markdown_table(lang_rows, "Per-Language Breakdown")
    write_csv(lang_rows, OUTPUT_DIR / "per_language_breakdown.csv")

    # 3. Delta decomposition
    deltas = compute_deltas(data)
    for arch in ["mT5", "ByT5"]:
        if arch in deltas:
            print_delta_table(deltas[arch], arch)

    # 4. Save full delta analysis
    delta_path = OUTPUT_DIR / "delta_analysis.json"
    with open(delta_path, "w", encoding="utf-8") as f:
        json.dump(deltas, f, indent=2)
    print(f"\nSaved delta analysis to {delta_path}")

    # 5. Save all tables as JSON
    tables_path = OUTPUT_DIR / "all_tables.json"
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": overall_rows,
            "per_language": lang_rows,
            "deltas": deltas,
        }, f, indent=2)
    print(f"Saved all tables to {tables_path}")


if __name__ == "__main__":
    main()
