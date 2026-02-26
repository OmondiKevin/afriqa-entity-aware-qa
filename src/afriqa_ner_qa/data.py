from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk


def load_afriqa_multi_config(
    dataset_name: str,
    configs: List[str],
    cache_dir: str | None = None,
) -> DatasetDict:
    """Load AfriQA by explicitly loading each language config then concatenating splits."""
    merged: dict[str, list[Dataset]] = {"train": [], "validation": [], "test": []}

    for cfg in configs:
        ds = load_dataset(dataset_name, cfg, cache_dir=cache_dir, trust_remote_code=True)
        # normalize split names if needed
        if "dev" in ds and "validation" not in ds:
            ds = DatasetDict({"train": ds["train"], "validation": ds["dev"], "test": ds["test"]})

        # inject lang column so merged DatasetDict has non-empty lang per example
        ds = DatasetDict(
            {split: ds[split].add_column("lang", [cfg] * ds[split].num_rows) for split in ds.keys()}
        )

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


def load_from_disk_datasetdict(path: str) -> DatasetDict:
    """Load a DatasetDict from disk (saved via save_to_disk)."""
    return load_from_disk(path)


def normalize_afriqa_answer(example: Dict[str, Any]) -> str:
    """Extract a single string answer from AfriQA example with robust fallbacks."""
    # Try answer fields in order: answer_lang (gold-passages), answer, answers
    for key in ("answer_lang", "answer"):
        ans = example.get(key)
        if ans is not None and isinstance(ans, str) and ans.strip():
            return ans.strip()

    answers = example.get("answers")
    if answers is None:
        return ""

    if isinstance(answers, str):
        s = answers.strip()
        if not s:
            return ""
        # Handle string like "['Emukwai']" or "[yes]"
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner.startswith("'") and inner.endswith("'"):
                return inner[1:-1].strip()
            if inner.startswith('"') and inner.endswith('"'):
                return inner[1:-1].strip()
            if "," in inner:
                first = inner.split(",")[0].strip().strip("'\"")
                return first
            return inner.strip("'\"")
        return s

    if isinstance(answers, list):
        if not answers:
            return ""
        first = answers[0]
        if isinstance(first, dict) and "text" in first:
            return str(first["text"]).strip()
        return str(first).strip()

    if isinstance(answers, dict) and "text" in answers:
        return str(answers["text"]).strip()

    return str(answers).strip()


def export_seq2seq_jsonl(
    ds: DatasetDict,
    out_dir: Path,
    question_field: str | None = None,
    lang_field: str | None = None,
    id_field: str = "id",
    prompt_prefix: str = "question: ",
    logger: Any = None,
) -> Dict[str, int]:
    """Write train/validation/test JSONL with id, lang, input_text, target_text. Returns counts per split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    total_dropped = 0
    q_field = question_field or "question"
    l_field = lang_field or "lang"
    # AfriQA gold-passages uses question_lang; base AfriQA uses question
    question_candidates = ("question_lang", "question") if q_field == "question" else (q_field,)

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        d = ds[split]
        out_path = out_dir / f"{split}.jsonl"
        written = 0
        dropped = 0
        with out_path.open("w", encoding="utf-8") as f:
            for i, row in enumerate(d):
                question = ""
                for qf in question_candidates:
                    q = row.get(qf)
                    if q is not None and (isinstance(q, str) and q.strip() or isinstance(q, list) and q):
                        question = q[0] if isinstance(q, list) else q
                        question = str(question).strip()
                        break
                target_text = str(normalize_afriqa_answer(row)).strip()
                input_text = f"{prompt_prefix}{question}".strip() if question else ""
                ex_id = str(row.get(id_field, i))
                lang = row.get(l_field) or ""
                if isinstance(lang, list):
                    lang = lang[0] if lang else ""
                lang = str(lang).strip()

                if not input_text or not target_text:
                    dropped += 1
                    if logger:
                        logger.warning(f"Dropped row (empty input or target): id={ex_id}")
                    continue

                obj = {
                    "id": ex_id,
                    "lang": lang,
                    "input_text": input_text,
                    "target_text": target_text,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
        counts[split] = written
        total_dropped += dropped


    if logger and total_dropped > 0:
        logger.warning(f"Dropped {total_dropped} rows in total (empty input_text or target_text)")

    return counts


def export_ner_seq2seq_jsonl(
    ds: DatasetDict,
    out_dir: Path,
    id_field: str = "id",
    prompt_prefix: str = "extract entities: ",
    logger: Any = None,
) -> Dict[str, int]:
    """Write train/validation/test JSONL for NER. Returns counts per split.
    Formats MasakhaNER tags into a sequence: 'PER: name, LOC: name'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    
    # MasakhaNER 2.0 Tags: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-DATE', 'I-DATE']
    # 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC, 7=B-DATE, 8=I-DATE
    tag_map = {1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC", 7: "DATE", 8: "DATE"}

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        d = ds[split]
        out_path = out_dir / f"{split}.jsonl"
        written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for i, row in enumerate(d):
                tokens = row["tokens"]
                ner_tags = row["ner_tags"]
                
                input_text = f"{prompt_prefix}{' '.join(tokens)}"
                entities = []
                current_entity = []
                current_type = None

                for token, tag_id in zip(tokens, ner_tags):
                    if tag_id == 0:
                        if current_entity:
                            entities.append(f"{current_type}: {' '.join(current_entity)}")
                            current_entity = []
                            current_type = None
                    else:
                        tag_type = tag_map.get(tag_id)
                        # If B- tag or type changed
                        if tag_id % 2 != 0 or tag_type != current_type:
                            if current_entity:
                                entities.append(f"{current_type}: {' '.join(current_entity)}")
                            current_entity = [token]
                            current_type = tag_type
                        else: # I- tag
                            current_entity.append(token)
                
                if current_entity:
                    entities.append(f"{current_type}: {' '.join(current_entity)}")

                target_text = ", ".join(entities) if entities else "none"

                ex_id = str(row.get(id_field, f"ner_{i}"))
                lang = row.get("lang", "unknown")
                if isinstance(lang, list):
                    lang = lang[0] if lang else "unknown"
                lang = str(lang).strip()

                obj = {
                    "id": ex_id,
                    "lang": lang,
                    "input_text": input_text,
                    "target_text": target_text,
                    "task": "ner",
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
        counts[split] = written

    return counts

    