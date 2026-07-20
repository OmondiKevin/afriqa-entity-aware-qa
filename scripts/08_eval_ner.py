#!/usr/bin/env python3
"""Evaluate NER performance from multitask prediction files.

Loads multitask prediction JSONL files, filters to NER examples
(lang == "unknown"), and computes:
  - Entity-level exact match precision, recall, F1
  - Per-entity-type breakdown (PER, LOC, ORG, DATE)
  - Comparison between mT5 and ByT5 NER quality
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

PREDICTIONS_DIR = Path("outputs_colab/predictions")
OUTPUT_DIR = Path("outputs/analysis")

CONFIGS = [
    ("multitask_mt5_test", "Multitask mT5 (Full)"),
    ("multitask_byt5_test", "Multitask ByT5 (Full)"),
    ("multitask_mt5_lora_test", "LoRA mT5"),
    ("multitask_byt5_lora_test", "LoRA ByT5"),
]

ENTITY_TYPES = ["PER", "LOC", "ORG", "DATE"]


def parse_entities(text: str) -> list[tuple[str, str]]:
    """Parse NER-formatted text into list of (entity_type, entity_value) tuples.

    Actual data format: "PER: Nyerere, LOC: Tanzania, ORG: TANU"
    Also handles: pipe-separated and bracket formats as fallbacks.
    Special case: "none" means no entities.
    """
    text = text.strip()
    if not text or text.lower() == "none":
        return []

    entities = []

    # Use regex to find all "TYPE: value" patterns globally.
    # Each entity runs from "TYPE:" to just before the next "TYPE:" or end of string.
    pattern = r"(PER|LOC|ORG|DATE|MISC)\s*:\s*"
    splits = re.split(pattern, text, flags=re.IGNORECASE)
    # splits will be: [prefix, TYPE1, value1, TYPE2, value2, ...]
    # Start from index 1, stepping by 2
    for i in range(1, len(splits) - 1, 2):
        etype = splits[i].upper()
        evalue = splits[i + 1].strip().rstrip(",").strip()
        if evalue:
            entities.append((etype, evalue))

    return entities


def normalize_entity(text: str) -> str:
    """Normalize entity text for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def compute_entity_metrics(
    gold_entities: list[tuple[str, str]],
    pred_entities: list[tuple[str, str]],
) -> dict:
    """Compute entity-level precision, recall, F1.

    An entity is considered correct if both type and normalized value match exactly.
    """
    gold_set = {(t, normalize_entity(v)) for t, v in gold_entities}
    pred_set = {(t, normalize_entity(v)) for t, v in pred_entities}

    if not gold_set and not pred_set:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_ner_predictions(stem: str) -> list[dict]:
    """Load NER-only rows from prediction JSONL."""
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
            if row.get("lang", "") == "unknown":
                rows.append(row)
    return rows


def analyze_config(rows: list[dict]) -> dict:
    """Analyze NER performance for a single config."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    n_parseable_gold = 0
    n_parseable_pred = 0
    n_empty_pred = 0

    for row in rows:
        gold_text = row.get("target_text", "")
        pred_text = row.get("prediction_text", "")

        gold_entities = parse_entities(gold_text)
        pred_entities = parse_entities(pred_text)

        if gold_entities:
            n_parseable_gold += 1
        if pred_entities:
            n_parseable_pred += 1
        if not pred_text.strip():
            n_empty_pred += 1

        metrics = compute_entity_metrics(gold_entities, pred_entities)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        # Per-type aggregation
        gold_by_type = defaultdict(set)
        pred_by_type = defaultdict(set)
        for t, v in gold_entities:
            gold_by_type[t].add(normalize_entity(v))
        for t, v in pred_entities:
            pred_by_type[t].add(normalize_entity(v))

        for etype in ENTITY_TYPES:
            g = gold_by_type.get(etype, set())
            p = pred_by_type.get(etype, set())
            per_type[etype]["tp"] += len(g & p)
            per_type[etype]["fp"] += len(p - g)
            per_type[etype]["fn"] += len(g - p)

    # Aggregate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_type_metrics = {}
    for etype in ENTITY_TYPES:
        tp = per_type[etype]["tp"]
        fp = per_type[etype]["fp"]
        fn = per_type[etype]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_type_metrics[etype] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f,
            "support": tp + fn,
        }

    return {
        "n_total": len(rows),
        "n_parseable_gold": n_parseable_gold,
        "n_parseable_pred": n_parseable_pred,
        "n_empty_pred": n_empty_pred,
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_type": per_type_metrics,
    }


def print_results(results: dict) -> None:
    print("\n" + "=" * 80)
    print("NER EVALUATION RESULTS")
    print("=" * 80)

    for stem, label in CONFIGS:
        if stem not in results:
            print(f"\n{label}: No predictions found")
            continue

        r = results[stem]
        o = r["overall"]
        print(f"\n### {label}")
        print(f"  Total NER examples: {r['n_total']}")
        print(f"  Parseable gold: {r['n_parseable_gold']}, Parseable pred: {r['n_parseable_pred']}, Empty pred: {r['n_empty_pred']}")
        print(f"  Overall: P={o['precision']:.3f}  R={o['recall']:.3f}  F1={o['f1']:.3f}  (TP={o['tp']} FP={o['fp']} FN={o['fn']})")
        print()

        print("  | Type | Support | TP | FP | FN | Precision | Recall | F1 |")
        print("  |------|--------:|---:|---:|---:|----------:|-------:|---:|")
        for etype in ENTITY_TYPES:
            t = r["per_type"].get(etype, {})
            if t.get("support", 0) == 0 and t.get("fp", 0) == 0:
                continue
            print(f"  | {etype} | {t.get('support', 0)} | {t.get('tp', 0)} | {t.get('fp', 0)} | {t.get('fn', 0)} | "
                  f"{t.get('precision', 0):.3f} | {t.get('recall', 0):.3f} | {t.get('f1', 0):.3f} |")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for stem, label in CONFIGS:
        rows = load_ner_predictions(stem)
        if rows:
            results[stem] = analyze_config(rows)
            print(f"{label}: {len(rows)} NER rows analyzed")
        else:
            print(f"{label}: No NER predictions found ({stem})")

    print_results(results)

    # Save JSON
    json_path = OUTPUT_DIR / "ner_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved NER evaluation to {json_path}")


if __name__ == "__main__":
    main()
