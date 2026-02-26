from __future__ import annotations

import argparse
import random
from pathlib import Path

from afriqa_ner_qa.config import load_config
from afriqa_ner_qa.data import (
    load_from_disk_datasetdict,
    export_seq2seq_jsonl,
    summarize_splits,
)
from afriqa_ner_qa.logging_utils import setup_logger
from afriqa_ner_qa.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure()

    seed = cfg.get("project", {}).get("seed", 42)
    random.seed(seed)

    logger = setup_logger(log_file=str(paths.outputs / "logs" / "01_prepare_qa_data.log"))

    afriqa_disk_path = cfg["data"]["afriqa_disk_path"]
    qa_seq2seq_out_dir = cfg["data"]["qa_seq2seq_out_dir"]

    logger.info(f"Loading AfriQA from disk: {afriqa_disk_path}")
    ds = load_from_disk_datasetdict(afriqa_disk_path)
    logger.info(f"AfriQA splits: {summarize_splits(ds)}")

    out_dir = Path(qa_seq2seq_out_dir)
    prompt_prefix = cfg.get("model", {}).get("prompt_prefix", "question: ")
    logger.info(f"Exporting seq2seq JSONL to: {out_dir} (prompt_prefix={repr(prompt_prefix)})")
    counts = export_seq2seq_jsonl(ds, out_dir, prompt_prefix=prompt_prefix, logger=logger)

    for split, count in counts.items():
        logger.info(f"  {split}: {count} examples")

    logger.info("Done.")


if __name__ == "__main__":
    main()
