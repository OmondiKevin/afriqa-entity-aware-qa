from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict

from afriqa_ner_qa.config import load_config
from afriqa_ner_qa.logging_utils import setup_logger
from afriqa_ner_qa.paths import ProjectPaths
from afriqa_ner_qa.data import (
    load_afriqa_multi_config,
    load_masakhaner_multi_config,
    summarize_splits,
)


def save_dataset_dict(ds: DatasetDict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure()

    logger = setup_logger(log_file=str(paths.outputs / "logs" / "00_download_and_subset.log"))
    afriqa_configs = cfg["data"]["afriqa_configs"]
    masakhaner_configs = cfg["data"]["masakhaner_configs"]
    cache_dir = cfg["data"].get("cache_dir")

    afriqa_name = cfg["data"]["afriqa_name"]
    masakhaner_name = cfg["data"]["masakhaner_name"]

    logger.info(f"AfriQA configs: {afriqa_configs}")
    logger.info(f"MasakhaNER configs: {masakhaner_configs}")
    logger.info(f"HF cache_dir: {cache_dir}")
    logger.info(f"Loading AfriQA: {afriqa_name} (configs: {afriqa_configs})")
    afriqa = load_afriqa_multi_config(afriqa_name, afriqa_configs, cache_dir=cache_dir)
    logger.info(f"AfriQA splits: {summarize_splits(afriqa)}")

    afriqa_out = paths.data_processed / "afriqa_swa_hau_yor"
    logger.info(f"Saving AfriQA subset to: {afriqa_out}")
    save_dataset_dict(afriqa, afriqa_out)

    logger.info(f"Loading MasakhaNER 2.0: {masakhaner_name} (configs: {masakhaner_configs})")
    ner = load_masakhaner_multi_config(masakhaner_name, masakhaner_configs, cache_dir=cache_dir)
    logger.info(f"MasakhaNER splits: {summarize_splits(ner)}")

    ner_out = paths.data_processed / "masakhaner2_swa_hau_yor"
    logger.info(f"Saving MasakhaNER subset to: {ner_out}")
    save_dataset_dict(ner, ner_out)

    logger.info("Done.")


if __name__ == "__main__":
    main()
