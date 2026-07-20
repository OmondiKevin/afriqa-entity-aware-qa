#!/usr/bin/env python3
"""Error analysis across all experiment configurations.

Loads prediction JSONL files, categorizes errors, and produces:
  - Per-config error type distribution
  - Per-language error breakdown
  - Qualitative examples (5 per category per language)
  - Markdown report
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PREDICTIONS_DIR = Path("outputs_colab/predictions")
OUTPUT_DIR = Path("outputs/analysis")

CONFIGS = [
    ("baseline_mt5_test", "Baseline mT5 (1x)"),
    ("matchedqa_mt5_test", "Matched-Vol mT5 (20x)"),
    ("matchedqa_byt5_test", "Matched-Vol ByT5 (20x)"),
    ("multitask_mt5_test", "Multitask mT5"),
    ("multitask_byt5_test", "Multitask ByT5"),
    ("multitask_mt5_lora_test", "LoRA mT5"),
    ("multitask_byt5_lora_test", "LoRA ByT5"),
]

LANGUAGES = ["swa", "hau", "yor"]
NER_TAGS = ["PER:", "LOC:", "ORG:", "DATE:"]


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def has_ner_format_leakage(text: str) -> bool:
    """Check if prediction contains NER formatting tags."""
    return any(tag in text for tag in NER_TAGS)


def categorize_error(pred: str, gold: str) -> str:
    """Categorize a prediction into error types."""
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)

    if not pred_norm:
        return "empty"
    if pred_norm == gold_norm:
        return "exact_match"
    if has_ner_format_leakage(pred):
        return "ner_leakage"

    f1 = token_f1(pred, gold)
    if f1 > 0.5:
        return "high_partial"  # >50% token overlap
    if f1 > 0:
        return "low_partial"   # some overlap but <50%
    return "wrong"  # zero overlap


def load_predictions(stem: str) -> list[dict]:
    """Load prediction JSONL, filtering to QA-only rows."""
    path = PREDICTIONS_DIR / f"{stem}.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Filter out NER rows
            if row.get("lang", "") == "unknown":
                continue
            rows.append(row)
    return rows


def analyze_config(rows: list[dict]) -> dict[str, Any]:
    """Analyze error distribution for a single config."""
    overall = defaultdict(int)
    by_lang = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(lambda: defaultdict(list))  # category -> lang -> examples

    for row in rows:
        pred = row.get("prediction_text", "")
        gold = row.get("target_text", "")
        lang = row.get("lang", "unknown")
        category = categorize_error(pred, gold)

        overall[category] += 1
        by_lang[lang][category] += 1

        # Collect examples (max 5 per category per language)
        if len(examples[category][lang]) < 5:
            examples[category][lang].append({
                "id": row.get("id", ""),
                "input": row.get("input_text", "")[:120],
                "gold": gold,
                "pred": pred,
                "f1": round(token_f1(pred, gold), 3),
            })

    n = len(rows)
    return {
        "n": n,
        "overall": {k: {"count": v, "pct": v / n if n else 0} for k, v in sorted(overall.items())},
        "by_lang": {
            lang: {k: {"count": v, "pct": v / by_lang[lang]["exact_match"] * 0 + v / sum(by_lang[lang].values()) if sum(by_lang[lang].values()) else 0}
                   for k, v in sorted(by_lang[lang].items())}
            for lang in sorted(by_lang.keys())
        },
        "examples": {cat: dict(langs) for cat, langs in examples.items()},
    }


def generate_markdown_report(results: dict[str, dict], output_path: Path) -> None:
    """Generate a markdown report with error analysis."""
    lines = [
        "# Error Analysis Report",
        "",
        "## Overall Error Distribution",
        "",
    ]

    # Summary table
    categories = ["exact_match", "high_partial", "low_partial", "wrong", "empty", "ner_leakage"]
    cat_labels = {
        "exact_match": "Exact Match",
        "high_partial": "Partial (F1>0.5)",
        "low_partial": "Partial (F1≤0.5)",
        "wrong": "Wrong (F1=0)",
        "empty": "Empty Prediction",
        "ner_leakage": "NER Format Leakage",
    }

    header = "| Config | n |"
    for cat in categories:
        header += f" {cat_labels.get(cat, cat)} |"
    lines.append(header)
    sep = "|---|---:|"
    for _ in categories:
        sep += "---:|"
    lines.append(sep)

    for stem, label in CONFIGS:
        if stem not in results:
            continue
        r = results[stem]
        row = f"| {label} | {r['n']} |"
        for cat in categories:
            info = r["overall"].get(cat, {"count": 0, "pct": 0})
            row += f" {info['count']} ({info['pct']:.1%}) |"
        lines.append(row)
    lines.append("")

    # Per-language tables for top 3 configs
    lines.append("## Per-Language Error Breakdown")
    lines.append("")
    for stem, label in [("matchedqa_byt5_test", "Matched-Vol ByT5"), ("multitask_mt5_test", "Multitask mT5")]:
        if stem not in results:
            continue
        lines.append(f"### {label}")
        lines.append("")
        header = "| Language |"
        for cat in categories:
            header += f" {cat_labels.get(cat, cat)} |"
        lines.append(header)
        sep = "|---|"
        for _ in categories:
            sep += "---:|"
        lines.append(sep)

        r = results[stem]
        for lang in LANGUAGES:
            lang_data = r.get("by_lang", {}).get(lang, {})
            row = f"| {lang.upper()} |"
            for cat in categories:
                info = lang_data.get(cat, {"count": 0, "pct": 0})
                row += f" {info['count']} ({info['pct']:.1%}) |"
            lines.append(row)
        lines.append("")

    # Qualitative examples (best model only)
    best_stem = "matchedqa_byt5_test"
    if best_stem in results:
        lines.append("## Qualitative Examples (Matched-Vol ByT5)")
        lines.append("")
        for cat in categories:
            examples = results[best_stem].get("examples", {}).get(cat, {})
            if not examples:
                continue
            lines.append(f"### {cat_labels.get(cat, cat)}")
            lines.append("")
            for lang in LANGUAGES:
                lang_examples = examples.get(lang, [])
                if not lang_examples:
                    continue
                lines.append(f"**{lang.upper()}:**")
                lines.append("")
                for ex in lang_examples[:3]:
                    lines.append(f"- **Gold:** {ex['gold']}")
                    lines.append(f"  **Pred:** {ex['pred']}")
                    lines.append(f"  *(F1={ex['f1']})*")
                    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for stem, label in CONFIGS:
        rows = load_predictions(stem)
        if rows:
            results[stem] = analyze_config(rows)
            print(f"{label}: {len(rows)} QA rows analyzed")
        else:
            print(f"{label}: No predictions found ({stem})")

    # Save raw analysis JSON
    json_path = OUTPUT_DIR / "error_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved error analysis JSON to {json_path}")

    # Generate markdown report
    md_path = OUTPUT_DIR / "error_analysis_report.md"
    generate_markdown_report(results, md_path)
    print(f"Saved error analysis report to {md_path}")


if __name__ == "__main__":
    main()
