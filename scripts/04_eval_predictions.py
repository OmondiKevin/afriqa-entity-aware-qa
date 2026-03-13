from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from afriqa_ner_qa.config import load_config
from afriqa_ner_qa.eval import evaluate_predictions
from afriqa_ner_qa.logging_utils import setup_logger
from afriqa_ner_qa.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pred_path", default=None, help="Override predictions file path")
    parser.add_argument("--log_path", default=None, help="Override evaluation log file path")
    parser.add_argument("--qa_only", action="store_true", help="Filter out NER examples (lang=unknown) before evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure()

    (paths.outputs / "metrics").mkdir(parents=True, exist_ok=True)

    default_log_path = paths.outputs / "logs" / "04_eval_predictions.log"
    log_path = Path(args.log_path) if args.log_path else default_log_path
    logger = setup_logger(log_file=str(log_path))

    run_cfg = cfg.get("run", {})
    eval_cfg = cfg.get("eval", {})

    pred_path = args.pred_path or run_cfg.get("baseline_pred_path", "outputs/predictions/baseline_mt5_test.jsonl")
    pred_path = Path(pred_path)

    if not pred_path.exists():
        logger.error(f"Predictions file not found: {pred_path}")
        return

    logger.info(f"Loading predictions from {pred_path}")
    rows: list[dict] = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("lang", "").strip():
                row["lang"] = "unknown"
            rows.append(row)

    logger.info(f"Loaded {len(rows)} rows from {pred_path}")

    if args.qa_only:
        rows = [r for r in rows if r.get("lang") != "unknown"]
        logger.info(f"Filtered to {len(rows)} QA-only rows (--qa_only passed)")
        if not rows:
            logger.warning("No rows remain after QA-only filtering")
            return

    do_semantic = eval_cfg.get("do_semantic", False)
    labse_model = eval_cfg.get("labse_model", "sentence-transformers/LaBSE")
    semantic_batch_size = eval_cfg.get("semantic_batch_size", 32)

    if do_semantic:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            logger.warning("sentence-transformers not installed; skipping semantic similarity (EM/F1 only)")
            do_semantic = False

    strip_tags = eval_cfg.get("strip_ner_tags", False)

    metrics = evaluate_predictions(
        rows,
        do_semantic=do_semantic,
        labse_model=labse_model,
        batch_size=semantic_batch_size,
        strip_tags=strip_tags,
        logger=logger,
    )

    if args.pred_path:
        stem = pred_path.stem
        if args.qa_only and not stem.endswith("_qa_only"):
            stem += "_qa_only"
        metrics_json_path = paths.outputs / "metrics" / f"{stem}_metrics.json"
        metrics_csv_path = paths.outputs / "metrics" / f"{stem}_metrics.csv"
    else:
        json_path = run_cfg.get("baseline_metrics_json", "outputs/metrics/baseline_mt5_test_metrics.json")
        csv_path = run_cfg.get("baseline_metrics_csv", "outputs/metrics/baseline_mt5_test_metrics.csv")
        if args.qa_only:
            json_path = json_path.replace(".json", "_qa_only.json")
            csv_path = csv_path.replace(".csv", "_qa_only.csv")
        metrics_json_path = Path(json_path)
        metrics_csv_path = Path(csv_path)

    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Wrote metrics JSON to {metrics_json_path}")

    csv_columns = ["group", "n", "em", "f1"]
    if "semantic" in metrics.get("overall", {}):
        csv_columns.append("semantic")

    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        overall_row = {"group": "overall", **metrics["overall"]}
        writer.writerow(overall_row)
        for lang, stats in metrics.get("per_lang", {}).items():
            writer.writerow({"group": f"lang:{lang}", **stats})

    logger.info(f"Wrote metrics CSV to {metrics_csv_path}")

    # Compact summary
    o = metrics["overall"]
    logger.info(f"Overall: n={o['n']} EM={o['em']:.4f} F1={o['f1']:.4f}" + (f" semantic={o.get('semantic', 0):.4f}" if "semantic" in o else ""))
    for lang, stats in metrics.get("per_lang", {}).items():
        s = f"  {lang}: n={stats['n']} EM={stats['em']:.4f} F1={stats['f1']:.4f}"
        if "semantic" in stats:
            s += f" semantic={stats['semantic']:.4f}"
        logger.info(s)

    logger.info("Done.")


if __name__ == "__main__":
    main()
