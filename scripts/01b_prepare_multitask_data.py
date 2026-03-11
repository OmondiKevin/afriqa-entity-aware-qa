from __future__ import annotations

import argparse
import random
import json
from pathlib import Path

from afriqa_ner_qa.config import load_config
from afriqa_ner_qa.data import (
    load_from_disk_datasetdict,
    export_seq2seq_jsonl,
    export_ner_seq2seq_jsonl,
    summarize_splits,
)
from afriqa_ner_qa.logging_utils import setup_logger
from afriqa_ner_qa.paths import ProjectPaths

def combine_and_shuffle_jsonl(file1: Path, file2: Path, out_file: Path, seed: int = 42, upsample_file1: int = 1) -> int:
    lines = []
    if file1.exists():
        with file1.open("r", encoding="utf-8") as f:
            f1_lines = f.readlines()
            for _ in range(upsample_file1):
                lines.extend(f1_lines)
    if file2.exists():
        with file2.open("r", encoding="utf-8") as f:
            lines.extend(f.readlines())
    
    random.seed(seed)
    random.shuffle(lines)
    
    with out_file.open("w", encoding="utf-8") as f:
        f.writelines(lines)
            
    return len(lines)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure()

    seed = cfg.get("project", {}).get("seed", 42)
    random.seed(seed)

    logger = setup_logger(log_file=str(paths.outputs / "logs" / "01b_prepare_multitask_data.log"))

    afriqa_disk_path = cfg["data"]["afriqa_disk_path"]
    ner_disk_path = paths.data_processed / "masakhaner2_swa_hau_yor"

    logger.info(f"Loading AfriQA from disk: {afriqa_disk_path}")
    afriqa_ds = load_from_disk_datasetdict(str(afriqa_disk_path))
    logger.info(f"Loading MasakhaNER from disk: {ner_disk_path}")
    ner_ds = load_from_disk_datasetdict(str(ner_disk_path))

    qa_out_dir = paths.data_processed / "qa_temp"
    ner_out_dir = paths.data_processed / "ner_temp"
    final_out_dir = paths.data_processed / (Path(cfg["data"]["qa_seq2seq_out_dir"]).name + "_multitask")

    prompt_prefix_qa = cfg.get("model", {}).get("prompt_prefix", "question: ")
    logger.info(f"Exporting QA seq2seq JSONL to: {qa_out_dir}")
    export_seq2seq_jsonl(afriqa_ds, str(qa_out_dir), prompt_prefix=prompt_prefix_qa, logger=logger)
    
    prompt_prefix_ner = cfg.get("multitask", {}).get("ner_prompt_prefix", "extract entities: ")
    logger.info(f"Exporting NER seq2seq JSONL to: {ner_out_dir}")
    export_ner_seq2seq_jsonl(ner_ds, str(ner_out_dir), prompt_prefix=prompt_prefix_ner, logger=logger)

    final_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Combining and shuffling into: {final_out_dir}")
    
    qa_upsample_factor = cfg.get("multitask", {}).get("qa_upsample_factor", 1)
    
    for split in ["train", "validation", "test"]:
        file1 = qa_out_dir / f"{split}.jsonl"
        file2 = ner_out_dir / f"{split}.jsonl"
        out_file = final_out_dir / f"{split}.jsonl"
        factor = qa_upsample_factor if split == "train" else 1
        count = combine_and_shuffle_jsonl(file1, file2, out_file, seed=seed, upsample_file1=factor)
        logger.info(f"  {split}: {count} total examples combined (QA upsampled {factor}x)")

    logger.info("Done.")

if __name__ == "__main__":
    main()
