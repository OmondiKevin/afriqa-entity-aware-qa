from __future__ import annotations

from typing import List, Dict, Any, Tuple
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def load_afriqa_multi_config(
    dataset_name: str,
    configs: List[str],
    cache_dir: str | None = None,
) -> DatasetDict:
    """Load AfriQA by explicitly loading each language config then concatenating splits."""
    merged: dict[str, list[Dataset]] = {"train": [], "validation": [], "test": []}

    for cfg in configs:
        ds = load_dataset(dataset_name, cfg, cache_dir=cache_dir)
        # normalize split names if needed
        if "dev" in ds and "validation" not in ds:
            ds = DatasetDict({"train": ds["train"], "validation": ds["dev"], "test": ds["test"]})

        for split in merged.keys():
            if split in ds:
                merged[split].append(ds[split])

    out = DatasetDict()
    for split, parts in merged.items():
        if parts:
            out[split] = concatenate_datasets(parts)
    return out


def load_masakhaner(dataset_name: str, cache_dir: str | None = None) -> DatasetDict:
    return load_dataset(dataset_name, cache_dir=cache_dir)


def load_masakhaner_multi_config(
    dataset_name: str,
    configs: List[str],
    cache_dir: str | None = None,
) -> DatasetDict:
    """Load MasakhaNER by loading each language config then concatenating splits."""
    merged: dict[str, list[Dataset]] = {"train": [], "validation": [], "test": []}

    for cfg in configs:
        ds = load_dataset(dataset_name, cfg, cache_dir=cache_dir)
        if "dev" in ds and "validation" not in ds:
            ds = DatasetDict({"train": ds["train"], "validation": ds["dev"], "test": ds["test"]})

        for split in merged.keys():
            if split in ds:
                merged[split].append(ds[split])

    out = DatasetDict()
    for split, parts in merged.items():
        if parts:
            out[split] = concatenate_datasets(parts)
    return out


def filter_by_languages(ds: DatasetDict, lang_field: str, languages: List[str]) -> DatasetDict:
    languages_set = set(languages)

    def _keep(example: Dict[str, Any]) -> bool:
        return example.get(lang_field) in languages_set

    out = DatasetDict()
    for split, d in ds.items():
        out[split] = d.filter(_keep)
    return out


def summarize_splits(ds: DatasetDict) -> List[Tuple[str, int]]:
    return [(split, ds[split].num_rows) for split in ds.keys()]
    