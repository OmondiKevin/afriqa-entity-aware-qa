from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from datasets import Dataset


def _get_extra_id_bad_words_ids(tokenizer: Any) -> Optional[List[List[int]]]:
    """Build bad_words_ids to ban T5 sentinel tokens <extra_id_0> through <extra_id_99>."""
    bad_words_ids = []
    for i in range(100):
        sentinel = f"<extra_id_{i}>"
        ids = tokenizer.encode(sentinel, add_special_tokens=False)
        if ids:
            bad_words_ids.append(ids)
    return bad_words_ids if bad_words_ids else None


def _clean_extra_id_from_pred(text: str) -> str:
    """Remove <extra_id_N> substrings and strip extra punctuation/whitespace."""
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(".,;:!?")
    return text.strip()


def normalize_text(s: str) -> str:
    """Lowercase, strip, remove punctuation, collapse whitespace."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(pred: str, gold: str) -> int:
    """Return 1 if normalized pred equals normalized gold, else 0."""
    return 1 if normalize_text(pred) == normalize_text(gold) else 0


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 (SQuAD-style) on normalized whitespace tokens. Handle empty cases."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(
    rows: List[Dict[str, Any]],
    do_semantic: bool = False,
    labse_model: str = "sentence-transformers/LaBSE",
    batch_size: int = 32,
    logger: Any = None,
) -> Dict[str, Any]:
    """Compute EM, token-F1, and optional semantic similarity. Returns overall + per_lang metrics."""
    pred_col = "prediction_text"
    target_col = "target_text"
    lang_col = "lang"

    overall_em = 0.0
    overall_f1 = 0.0
    overall_semantic: List[float] = []
    by_lang: Dict[str, Dict[str, Any]] = {}

    pred_texts = [r.get(pred_col, "") or "" for r in rows]
    gold_texts = [r.get(target_col, "") or "" for r in rows]
    langs = [r.get(lang_col, "") or "unknown" for r in rows]

    for i, (pred, gold, lang) in enumerate(zip(pred_texts, gold_texts, langs)):
        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        overall_em += em
        overall_f1 += f1
        if lang not in by_lang:
            by_lang[lang] = {"n": 0, "em": 0.0, "f1": 0.0, "semantic": []}
        by_lang[lang]["n"] += 1
        by_lang[lang]["em"] += em
        by_lang[lang]["f1"] += f1

    n = len(rows)
    result: Dict[str, Any] = {
        "overall": {
            "n": n,
            "em": overall_em / n if n else 0.0,
            "f1": overall_f1 / n if n else 0.0,
        },
        "per_lang": {},
    }

    for lang, stats in by_lang.items():
        nn = stats["n"]
        result["per_lang"][lang] = {
            "n": nn,
            "em": stats["em"] / nn if nn else 0.0,
            "f1": stats["f1"] / nn if nn else 0.0,
        }
        if "semantic" in stats and stats["semantic"]:
            result["per_lang"][lang]["semantic"] = sum(stats["semantic"]) / len(stats["semantic"])

    if do_semantic:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer(labse_model)
            pred_embs = model.encode(pred_texts, batch_size=batch_size, show_progress_bar=False)
            gold_embs = model.encode(gold_texts, batch_size=batch_size, show_progress_bar=False)
            pred_embs = pred_embs / (np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-9)
            gold_embs = gold_embs / (np.linalg.norm(gold_embs, axis=1, keepdims=True) + 1e-9)
            sims = np.sum(pred_embs * gold_embs, axis=1)
            overall_semantic = float(np.mean(sims))
            result["overall"]["semantic"] = overall_semantic
            for lang in by_lang:
                idx = [i for i, l in enumerate(langs) if l == lang]
                if idx:
                    result["per_lang"][lang]["semantic"] = float(np.mean(sims[idx]))
        except Exception as e:
            if logger:
                logger.warning(f"Semantic similarity failed ({e}); continuing with EM/F1 only")

    return result


def _pad_batch(batch_input_ids: List[List[int]], batch_attention_mask: List[List[int]], pad_token_id: int) -> tuple:
    """Pad batch to max length and return (padded_input_ids, padded_attention_mask) as tensors."""
    import torch

    max_len = max(len(ids) for ids in batch_input_ids)
    pad_id = pad_token_id if pad_token_id is not None else 0
    padded_ids = []
    padded_mask = []
    for ids, mask in zip(batch_input_ids, batch_attention_mask):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_mask.append(mask + [0] * pad_len)
    return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(padded_mask, dtype=torch.long)


def _looks_weird_unicode(text: str) -> bool:
    """Heuristic: many non-ASCII or unusual chars suggests decoding issues."""
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text) > 0.3 or any(ord(c) < 32 and c not in "\t\n\r" for c in text)


def generate_predictions(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    max_new_tokens: int,
    device: Any,
    id_col: str = "id",
    lang_col: str = "lang",
    target_col: str = "target_text",
    input_text_col: str = "input_text",
    batch_size: int = 8,
    min_new_tokens: int = 0,
    num_beams: int = 4,
    debug_first_batch: bool = True,
    print_examples: int = 1,
    log_raw_on_weird: bool = False,
    log_raw_first_n: int = 0,
    allow_all_match: bool = False,
    skip_bad_words: bool = False,
    logger: Any = None,
) -> List[Dict[str, Any]]:
    """Generate predictions from a tokenized dataset. Dataset must have input_ids and attention_mask.
    Returns list of dicts with id, lang, input_text, prediction_text, target_text.
    """
    if "input_ids" not in dataset.column_names or "attention_mask" not in dataset.column_names:
        raise ValueError("Dataset must contain input_ids and attention_mask columns (use load_and_tokenize_jsonl_splits)")

    model.eval()
    results: List[Dict[str, Any]] = []
    n = len(dataset)
    logged_first = False

    for i in range(0, n, batch_size):
        batch = dataset[i : i + batch_size]
        batch_input_ids = batch["input_ids"]
        batch_attention_mask = batch["attention_mask"]
        if isinstance(batch_input_ids[0], int):
            batch_input_ids = [batch_input_ids]
            batch_attention_mask = [batch_attention_mask]

        input_ids, attention_mask = _pad_batch(
            batch_input_ids, batch_attention_mask, tokenizer.pad_token_id
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        bad_words_ids = None if skip_bad_words else _get_extra_id_bad_words_ids(tokenizer)

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "num_beams": num_beams,
            "do_sample": False,
        }
        if bad_words_ids is not None:
            gen_kwargs["bad_words_ids"] = bad_words_ids

        generated_ids = model.generate(**gen_kwargs)

        prediction_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        prediction_texts = [_clean_extra_id_from_pred(p) for p in prediction_texts]

        ids = batch.get(id_col, list(range(i, min(i + batch_size, n))))
        langs = batch.get(lang_col, [""] * len(batch_input_ids))
        targets = batch.get(target_col, [""] * len(batch_input_ids))
        input_texts = batch.get(input_text_col, [""] * len(batch_input_ids))
        if isinstance(ids, (int, str)):
            ids = [ids]
        if isinstance(langs, str):
            langs = [langs]
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        if debug_first_batch and not logged_first and logger:
            n_log = min(print_examples, len(prediction_texts))
            n_raw = min(log_raw_first_n, len(prediction_texts)) if log_raw_first_n > 0 else 0
            for j in range(n_log):
                decoded_input = tokenizer.batch_decode([batch_input_ids[j]], skip_special_tokens=True)[0]
                pred = prediction_texts[j].strip()
                tgt = targets[j] if j < len(targets) else ""
                if isinstance(tgt, str):
                    tgt = tgt.strip()
                else:
                    tgt = str(tgt).strip()
                logger.info(f"[example {j+1}] input: {decoded_input[:100]}...")
                logger.info(f"[example {j+1}] target (gold): {tgt}")
                logger.info(f"[example {j+1}] prediction: {pred}")
                if j < n_raw:
                    raw = tokenizer.decode(generated_ids[j], skip_special_tokens=False)
                    logger.info(f"[example {j+1}] raw (skip_special_tokens=False): {raw}")
                if j == 0 and n_raw > 0:
                    gen_ids = generated_ids[j]
                    if hasattr(gen_ids, "cpu"):
                        gen_ids = gen_ids.cpu().tolist()
                    elif hasattr(gen_ids, "tolist"):
                        gen_ids = gen_ids.tolist()
                    logger.info(f"[example 1] generated token ids: {gen_ids}")
                if log_raw_on_weird and _looks_weird_unicode(pred) and j >= n_raw:
                    raw = tokenizer.decode(generated_ids[j], skip_special_tokens=False)
                    logger.info(f"[example {j+1}] raw (skip_special_tokens=False): {raw}")
            logged_first = True

        for j in range(len(prediction_texts)):
            ex_id = ids[j] if j < len(ids) else i + j
            lang = langs[j] if j < len(langs) else ""
            inp = input_texts[j] if j < len(input_texts) else ""
            pred = prediction_texts[j].strip()
            tgt = targets[j] if j < len(targets) else ""
            if isinstance(tgt, str):
                tgt = tgt.strip()
            else:
                tgt = str(tgt).strip()
            row: Dict[str, Any] = {
                "id": ex_id,
                "lang": lang,
                "input_text": inp,
                "prediction_text": pred,
                "target_text": tgt,
            }
            if log_raw_first_n > 0 and (i + j) < log_raw_first_n:
                row["prediction_raw"] = tokenizer.decode(generated_ids[j], skip_special_tokens=False)
            results.append(row)

    # Guard against gold-copied predictions (skip in overfit mode where high match is expected)
    if not allow_all_match:
        n_equal = sum(1 for r in results if r["prediction_text"].strip() == r["target_text"].strip())
        if n_equal == len(results) and len(results) > 0:
            raise RuntimeError(
                f"All {len(results)} predictions match gold exactly (prediction_text == target_text). "
                "Predictions appear to be gold-copied; ensure prediction_text comes from model.generate outputs, not labels."
            )

    # Sanity: at least 90% of predictions contain at least one alphanumeric character (warning only, do not crash)
    n_readable = sum(1 for r in results if any(c.isalnum() for c in r["prediction_text"]))
    if len(results) > 0 and n_readable / len(results) < 0.9 and logger:
        logger.warning(
            f"Decoded predictions may be corrupted; only {n_readable}/{len(results)} ({100*n_readable/len(results):.1f}%) contain alphanumeric chars."
        )

    return results
