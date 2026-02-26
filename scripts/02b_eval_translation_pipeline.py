from __future__ import annotations

import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from afriqa_ner_qa.config import load_config
from afriqa_ner_qa.logging_utils import setup_logger
from afriqa_ner_qa.paths import ProjectPaths


def get_nllb_lang_code(lang: str) -> str:
    # Mapping for swa, hau, yor
    # Swahili: swh_Latn
    # Hausa: hau_Latn
    # Yoruba: yor_Latn
    mapping = {
        "swa": "swh_Latn",
        "hau": "hau_Latn",
        "yor": "yor_Latn",
    }
    return mapping.get(lang.lower(), "eng_Latn")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subset", type=int, default=0, help="Test on a subset of examples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure()

    logger = setup_logger(log_file=str(paths.outputs / "logs" / "02b_eval_translation_pipeline.log"))

    qa_seq2seq_dir = cfg["data"]["qa_seq2seq_out_dir"]
    data_files = {"test": str(Path(qa_seq2seq_dir) / "test.jsonl")}
    
    if not Path(data_files["test"]).exists():
        logger.error(f"Test data file not found: {data_files['test']}")
        return

    logger.info(f"Loading test split from {data_files['test']}")
    ds = load_dataset("json", data_files=data_files)["test"]

    if args.subset > 0:
        ds = ds.select(range(min(args.subset, len(ds))))
        logger.info(f"Using subset of {len(ds)} examples")

    use_mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if use_mps else "cpu")
    logger.info(f"Using device: {device}")

    # Load NLLB
    model_name = "facebook/nllb-200-distilled-600M"
    logger.info(f"Loading NLLB-200 translation model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    except Exception as e:
        logger.warning(f"Could not load NLLB model: {e}. Ensure transformers and pyarrow are working.")
        return

    # Load QA model 
    qa_model_name = "google/flan-t5-base"
    logger.info(f"Loading English QA model: {qa_model_name}")
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name).to(device)

    def translate(texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if not texts: return []
        tokenizer.src_lang = src_lang
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=64)
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def answer_english(questions: list[str]) -> list[str]:
        if not questions: return []
        inputs = qa_tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            generated_tokens = qa_model.generate(**inputs, max_new_tokens=32)
        return qa_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    batch_size = 16
    results = []
    
    output_path = paths.outputs / "predictions" / "translation_pipeline_test.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting translation pipeline inference...")
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i: i+batch_size]
        langs = batch["lang"]
        input_texts = batch["input_text"]
        
        # input_text usually has "question: " prefix. Let's strip it to translate just the question.
        clean_qs = [q.replace("question: ", "").strip() for q in input_texts]
        
        q_en = []
        for q, lang in zip(clean_qs, langs):
            src_lang = get_nllb_lang_code(lang)
            tgt_lang = "eng_Latn"
            ans = translate([q], src_lang, tgt_lang)[0]
            q_en.append(ans)
            
        a_en = answer_english(q_en)
        
        a_orig = []
        for a, lang in zip(a_en, langs):
            src_lang = "eng_Latn"
            tgt_lang = get_nllb_lang_code(lang)
            translated_ans = translate([a], src_lang, tgt_lang)[0]
            a_orig.append(translated_ans)
            
        for idx in range(len(batch["id"])):
            row = {
                "id": batch["id"][idx],
                "lang": batch["lang"][idx],
                "input_text": batch["input_text"][idx],
                "target_text": batch["target_text"][idx],
                "prediction_text": a_orig[idx],
                "intermediate_q_en": q_en[idx],
                "intermediate_a_en": a_en[idx],
            }
            results.append(row)
            
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    logger.info(f"Saved {len(results)} predictions to {output_path}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
