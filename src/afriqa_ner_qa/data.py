from __future__ import annotations

from typing import List, Dict, Any, Tuple
from datasets import DatasetDict, load_dataset


def load_afriqa(dataset_name: str, cache_dir: str | None = None) -> DatasetDict:
    return load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)


def load_masakhaner(dataset_name: str, cache_dir: str | None = None) -> DatasetDict:
    return load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)


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
    