from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_jsonl_split(data_dir: str | Path) -> DatasetDict:
    """Load train and validation JSONL from qa_seq2seq_out_dir."""
    data_dir = Path(data_dir)
    data_files = {
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "validation.jsonl"),
    }
    return load_dataset("json", data_files=data_files)


def load_and_tokenize_jsonl_splits(
    data_dir: str | Path,
    tokenizer: Any,
    max_source_length: int,
    max_target_length: int,
    logger: Any = None,
) -> DatasetDict:
    """Load train/validation/test JSONL, tokenize each split, return DatasetDict with input_ids, attention_mask, labels, id, lang, input_text, target_text."""
    data_dir = Path(data_dir)
    data_files = {}
    for split in ("train", "validation", "test"):
        p = data_dir / f"{split}.jsonl"
        if p.exists():
            data_files[split] = str(p)

    if not data_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")

    ds = load_dataset("json", data_files=data_files)

    def _tokenize(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        out = tokenize_function(
            examples,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        return out

    result = {}
    for split in ds:
        tokenized = ds[split].map(
            _tokenize,
            batched=True,
            desc=f"Tokenizing {split}",
        )
        result[split] = tokenized

    # Debug: log one example (decoded input, decoded label ignoring -100, count of -100 tokens)
    if logger and result:
        first_split = next(iter(result))
        ex = result[first_split][0]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
        label_ids = [l for l in labels if l != -100]
        decoded_label = tokenizer.decode(label_ids, skip_special_tokens=True) if label_ids else ""
        n_masked = sum(1 for l in labels if l == -100)
        logger.info(f"[tokenize debug] decoded input: {decoded_input[:120]}...")
        logger.info(f"[tokenize debug] decoded label (ignoring -100): {decoded_label}")
        logger.info(f"[tokenize debug] labels: -100 count={n_masked}, total={len(labels)}")

    return DatasetDict(result)


def tokenize_function(
    examples: Dict[str, List[Any]],
    tokenizer: Any,
    max_source_length: int,
    max_target_length: int,
    input_col: str = "input_text",
    target_col: str = "target_text",
) -> Dict[str, Any]:
    """Tokenize input_text and target_text for seq2seq. Explicitly mask pad tokens in labels with -100.
    Inputs: truncation to max_source_length. Targets: padding='max_length' for stable masking on CPU."""
    model_inputs = tokenizer(
        examples[input_col],
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    target_tokenized = tokenizer(
        examples[target_col],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model_inputs["labels"] = [
        [(tid if tid != pad_id else -100) for tid in label_ids]
        for label_ids in target_tokenized["input_ids"]
    ]
    return model_inputs


def build_seq2seq_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: Seq2SeqTrainingArguments,
    max_source_length: int,
    max_target_length: int,
) -> Seq2SeqTrainer:
    """Build Seq2SeqTrainer with tokenized datasets."""

    def _tokenize(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenize_function(
            examples,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

    tokenized_train = train_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    tokenized_eval = eval_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
