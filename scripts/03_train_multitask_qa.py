from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments

from afriqa_ner_qa.config import load_config
from afriqa_ner_qa.eval import _clean_extra_id_from_pred, exact_match, generate_predictions
from afriqa_ner_qa.logging_utils import setup_logger
from afriqa_ner_qa.paths import ProjectPaths
from afriqa_ner_qa.train import build_seq2seq_trainer, load_and_tokenize_jsonl_splits, load_jsonl_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predict_only", action="store_true", help="Skip training; load checkpoint and generate predictions only")
    parser.add_argument("--no_debug_predict_train", action="store_true",
                        help="Disable trainer.predict sanity check on train (enabled by default in overfit mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure()

    seed = cfg.get("project", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger = setup_logger(log_file=str(paths.outputs / "logs" / "03_train_multitask_qa.log"))

    qa_seq2seq_dir = str(paths.data_processed / (Path(cfg["data"]["qa_seq2seq_out_dir"]).name + "_multitask"))
    max_target_length = cfg["model"]["max_target_length"]
    eval_cfg = cfg.get("eval", {})
    generation_max_new_tokens = eval_cfg.get("generation_max_new_tokens", max_target_length)
    generation_min_new_tokens = eval_cfg.get("generation_min_new_tokens", 0)
    train_cfg = cfg.get("train", {})
    run_cfg = cfg.get("run", {})

    output_dir = run_cfg.get("multitask_output_dir", "outputs/checkpoints/multitask_mt5")
    pred_path = run_cfg.get("multitask_pred_path", "outputs/predictions/multitask_mt5_test.jsonl")
    overfit_pred_path = run_cfg.get("overfit_pred_path_mt", "outputs/predictions/overfit_mt5_multitask_train.jsonl")

    debug_cfg = cfg.get("debug", {})
    overfit_n = debug_cfg.get("overfit_n", 0)
    print_examples = debug_cfg.get("print_examples", 1)
    debug_predict_train = overfit_n > 0 and not args.no_debug_predict_train

    lora_cfg = cfg.get("lora", {})
    use_lora = lora_cfg.get("use_lora", False)
    if use_lora:
        output_dir = output_dir.replace("multitask_mt5", "multitask_mt5_lora")
        pred_path = pred_path.replace("multitask_mt5_test", "multitask_mt5_lora_test")

    model_cfg = cfg.get("model", {})
    if overfit_n > 0:
        model_name = "google/mt5-small"
        logger.info(f"Overfit mode: forcing model {model_name} (not {model_cfg.get('base', 'mt5-base')})")
    else:
        model_name = model_cfg.get("base", "google/mt5-base")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)

    use_mps = torch.backends.mps.is_available()
    use_cpu = use_mps or not torch.cuda.is_available()

    if not args.predict_only:
        max_source_length = train_cfg.get("max_source_length") or cfg["model"]["max_source_length"]
        if overfit_n > 0:
            max_target_length = train_cfg.get("debug_max_target_length", 16)
        else:
            max_target_length = train_cfg.get("max_target_length") or cfg["model"]["max_target_length"]
        logger.info(f"Tokenization: max_source_length={max_source_length}, max_target_length={max_target_length}")
        logger.info(f"Loading JSONL from {qa_seq2seq_dir}")
        ds = load_jsonl_split(qa_seq2seq_dir)
        train_ds = ds["train"]
        eval_ds = ds["validation"]

        if overfit_n > 0:
            train_ds = train_ds.select(range(min(overfit_n, len(train_ds))))
            eval_ds = train_ds  # evaluate and predict on same train slice
            logger.info(f"Overfit debug: train=eval=pred on same {len(train_ds)} examples")

        logger.info(f"Train: {len(train_ds)}, Validation: {len(eval_ds)}")

        logger.info(f"Loading model and tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_prefix_space=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError:
                logger.error("peft is not installed. Please install it using `pip install peft` or set use_lora: false.")
                sys.exit(1)
            
            lora_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("alpha", 32),
                target_modules=["q", "v"],
                lora_dropout=lora_cfg.get("dropout", 0.05),
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            model = get_peft_model(model, lora_config)
            logger.info("LoRA injected! Trainable parameters:")
            model.print_trainable_parameters()

        if overfit_n > 0:
            if not use_lora:
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            logger.info("Overfit mode: gradient checkpointing disabled, use_cache=True")
        else:
            if not use_lora:
                model.gradient_checkpointing_enable()
            model.config.use_cache = False
            logger.info("Adafactor + gradient checkpointing enabled for low-memory training")

        num_workers = train_cfg.get("num_workers", 0)
        pin_memory = not use_mps
        if use_mps:
            num_workers = 0
            pin_memory = False
            logger.info("MPS detected: using CPU to avoid OOM (dataloader_num_workers=0, pin_memory=False)")
        elif use_cpu:
            logger.info("Using CPU for training (no CUDA)")

        if overfit_n > 0:
            lr = float(debug_cfg.get("overfit_lr", 3e-4))
            logger.info(f"Overfit mode: learning_rate={lr:.2e} (from debug.overfit_lr)")
            weight_decay = 0.0
            warmup_steps = 0
            warmup_ratio = 0.0
            lr_scheduler_type = "constant"
            optim_name = "adamw_torch"
            overfit_max_steps_val = debug_cfg.get("overfit_max_steps", 0)
            epochs = debug_cfg.get("overfit_epochs", 50) if overfit_max_steps_val <= 0 else 1
        else:
            lr = float(train_cfg.get("lr", 3.0e-5))
            weight_decay = train_cfg.get("weight_decay", 0.01)
            warmup_steps = train_cfg.get("warmup_steps", 200)
            warmup_ratio = train_cfg.get("warmup_ratio", None)
            lr_scheduler_type = train_cfg.get("lr_scheduler_type", "linear")
            optim_name = "adafactor"
            epochs = train_cfg.get("epochs", 3)

        logger.info(f"Training: learning_rate={lr}, lr_scheduler_type={lr_scheduler_type}, optim={optim_name}")

        overfit_max_steps = debug_cfg.get("overfit_max_steps", 0) if overfit_n > 0 else 0  # 0 = use epochs
        per_device_bs = train_cfg.get("batch_size", 8) if overfit_n == 0 else 8
        grad_accum = 1 if overfit_n > 0 else train_cfg.get("grad_accum", 2)
        training_kw: dict = {
            "output_dir": output_dir,
            "eval_strategy": "steps",
            "eval_steps": train_cfg.get("eval_steps", 200) if overfit_n == 0 else 10,
            "save_strategy": "steps",
            "save_steps": train_cfg.get("save_steps", 200),
            "save_total_limit": train_cfg.get("save_total_limit", 1),
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "per_device_train_batch_size": per_device_bs,
            "per_device_eval_batch_size": per_device_bs,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "num_train_epochs": epochs,
            "fp16": train_cfg.get("fp16", False),
            "logging_steps": 1 if overfit_n > 0 else train_cfg.get("logging_steps", 50),
            "seed": seed,
            "dataloader_num_workers": num_workers,
            "dataloader_pin_memory": pin_memory,
            "optim": optim_name,
            "lr_scheduler_type": lr_scheduler_type,
            "predict_with_generate": False,
            "report_to": "none",
            "use_cpu": use_cpu,
            "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
            "label_smoothing_factor": 0.0,
        }
        if warmup_ratio is not None:
            training_kw["warmup_ratio"] = warmup_ratio
        else:
            training_kw["warmup_steps"] = warmup_steps

        if overfit_max_steps > 0:
            training_kw["max_steps"] = overfit_max_steps
            logger.info(f"Overfit mode: using max_steps={overfit_max_steps} (ignoring epochs)")

        training_args = Seq2SeqTrainingArguments(**training_kw)
        logger.info(f"max_grad_norm={training_args.max_grad_norm}")
        if overfit_n > 0:
            assert training_args.max_grad_norm > 0, "max_grad_norm must be > 0 for overfit mode"

        trainer = build_seq2seq_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            training_args=training_args,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

        logger.info("Starting training...")
        try:
            lr = float(trainer.args.learning_rate)
            lr_str = f"{lr:.2e}"
        except (TypeError, ValueError):
            lr_str = str(trainer.args.learning_rate)
        logger.info(f"Trainer args: learning_rate={lr_str}, lr_scheduler_type={trainer.args.lr_scheduler_type}, optim={trainer.args.optim} (constant scheduler => no decay)")
        try:
            trainer.train()
            trainer.save_model(output_dir)
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            sys.exit(1)

        # Sanity check: trainer.predict on train slice (overfit mode)
        if debug_predict_train:
            logger.info("Debug: running trainer.predict on train slice (predict_with_generate=True)")
            trainer.args.predict_with_generate = True
            trainer.args.generation_max_length = eval_cfg.get("generation_max_new_tokens", 16)
            trainer.args.generation_num_beams = 1
            pred_output = trainer.predict(trainer.eval_dataset)
            raw = pred_output.predictions
            if isinstance(raw, tuple):
                raw = raw[0]
            if hasattr(raw, "ndim"):
                if raw.ndim == 3:
                    token_ids = raw.argmax(-1)
                elif raw.ndim == 2:
                    token_ids = raw
                else:
                    raise ValueError(f"trainer.predict returned predictions with ndim={raw.ndim}, expected 2 (token_ids) or 3 (logits)")
            else:
                raise ValueError(f"trainer.predict returned unexpected type {type(raw)}, expected numpy/torch array")
            if hasattr(token_ids, "cpu"):
                token_ids = token_ids.cpu().numpy()
            if hasattr(token_ids, "astype"):
                token_ids = token_ids.astype("int64")
            decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            decoded = [_clean_extra_id_from_pred(p) for p in decoded]
            golds = train_ds["target_text"]
            inputs = train_ds["input_text"]
            n_log = min(print_examples, len(decoded))
            for j in range(n_log):
                inp = inputs[j] if j < len(inputs) else ""
                inp = inp[:100] + "..." if len(inp) > 100 else inp
                logger.info(f"[trainer.predict example {j+1}] input: {inp}")
                logger.info(f"[trainer.predict example {j+1}] gold: {golds[j] if j < len(golds) else ''}")
                logger.info(f"[trainer.predict example {j+1}] pred: {decoded[j] if j < len(decoded) else ''}")
        tokenizer.add_prefix_space = False
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved checkpoint to {output_dir}")
        model = trainer.model
    else:
        logger.info(f"Predict-only mode: loading best checkpoint from {output_dir}")
        
        if use_lora:
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_prefix_space=False)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = PeftModel.from_pretrained(base_model, output_dir)
            logger.info(f"Loaded base model {model_name} and LoRA adapters from {output_dir}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=False, add_prefix_space=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)

    # Generate predictions (from best checkpoint)
    max_source_length = train_cfg.get("max_source_length") or cfg["model"]["max_source_length"]
    if overfit_n > 0:
        max_target_length = train_cfg.get("debug_max_target_length", 16)
        gen_max_tokens = eval_cfg.get("generation_max_new_tokens", 16)
        gen_min_new_tokens = eval_cfg.get("generation_min_new_tokens", 1)
    else:
        max_target_length = train_cfg.get("max_target_length") or cfg["model"]["max_target_length"]
        gen_max_tokens = generation_max_new_tokens
        gen_min_new_tokens = generation_min_new_tokens

    model.config.use_cache = True
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    model = model.to(device)
    logger.info(f"Loading and tokenizing from {qa_seq2seq_dir} (max_source={max_source_length}, max_target={max_target_length})")
    tokenized_ds = load_and_tokenize_jsonl_splits(
        qa_seq2seq_dir,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        logger=logger,
    )

    if overfit_n > 0:
        pred_ds = tokenized_ds["train"].select(range(min(overfit_n, len(tokenized_ds["train"]))))
        out_path = Path(overfit_pred_path)
        logger.info(f"Overfit: evaluating on same {len(pred_ds)} train examples -> {out_path}")
    else:
        if "test" not in tokenized_ds:
            logger.warning("Test split not found, skipping predictions")
            eval_ds = None
        else:
            eval_ds = tokenized_ds["test"]
        out_path = Path(pred_path)

    pred_ds_final = pred_ds if overfit_n > 0 else eval_ds
    if pred_ds_final is not None:
        logger.info(f"Generating predictions on {len(pred_ds_final)} examples")
        preds = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            dataset=pred_ds_final,
            max_new_tokens=gen_max_tokens,
            min_new_tokens=gen_min_new_tokens,
            num_beams=1 if overfit_n > 0 else 4,
            device=device,
            debug_first_batch=True,
            print_examples=print_examples,
            log_raw_on_weird=overfit_n > 0,
            log_raw_first_n=5 if overfit_n > 0 else 0,
            allow_all_match=overfit_n > 0,
            skip_bad_words=overfit_n > 0,
            logger=logger,
        )
        # Proof-of-learning (overfit mode): print 5 examples and compute EM
        if overfit_n > 0 and preds:
            n_proof = min(5, len(preds))
            logger.info("=== PROOF-OF-LEARNING (overfit sanity) ===")
            for k in range(n_proof):
                p = preds[k]
                inp = p.get("input_text", "")[:100]
                gold = p.get("target_text", "")
                pred_clean = p.get("prediction_text", "")
                pred_raw = p.get("prediction_raw", "(not logged)")
                logger.info(f"[proof {k+1}] input: {inp}...")
                logger.info(f"[proof {k+1}] gold: {gold}")
                logger.info(f"[proof {k+1}] pred (clean): {pred_clean}")
                logger.info(f"[proof {k+1}] pred (raw): {pred_raw}")
            em_count = sum(exact_match(p.get("prediction_text", ""), p.get("target_text", "")) for p in preds)
            em_frac = em_count / len(preds) if preds else 0.0
            logger.info(f"=== Quick EM on {len(preds)} examples: {em_count}/{len(preds)} = {em_frac:.4f} ===")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for p in preds:
                out_row = {k: v for k, v in p.items() if k != "prediction_raw"}
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(preds)} predictions to {out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
