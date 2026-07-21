# Correction Audit: manuscript scientific inconsistencies
**Date:** 2026-07-21  
**Auditor:** Acting as senior ML researcher / ACL reviewer

---

## Issue 1: ByT5 Generation Limit — BLOCKER

### Evidence Inspected
- All configs: `eval.generation_max_new_tokens = 16`
- All ByT5 prediction files: max prediction length = **16 bytes exactly**
- All mT5 prediction files: max prediction length = 87 bytes (not capped)
- Gold answer length distribution: p75 = 16 bytes; **24.1% of gold answers exceed 16 bytes**

### Key Numbers
| Prediction File | max_pred_len | At 16 bytes | Truncated at 16 (pred=16, gold>16) |
|---|---:|---:|---:|
| matchedqa_byt5_test.jsonl | 16 | 202/848 (23.8%) | 86/848 (10.1%) |
| baseline_byt5_test_v2.jsonl | 16 | 71/848 (8.4%) | 17/848 (2.0%) |
| multitask_byt5_test.jsonl | 16 | 1395/6328 (22.0%) | 972/6328 (15.4%) |
| multitask_byt5_lora_test.jsonl | 16 | 2270/6328 (35.9%) | 1346/6328 (21.3%) |
| matchedqa_mt5_test.jsonl | 80 | 38/848 (4.5%) | 3/848 (0.4%) |
| baseline_mt5_test.jsonl | 87 | 32/848 (3.8%) | 6/848 (0.7%) |

**The 16-byte cap created a fundamentally unfair comparison.** ByT5 encodes each character as 1–4 bytes (UTF-8). For African-language tokens with diacritics (Yoruba, Hausa), a single character may be 2–3 bytes. The 16-byte limit that was adequate for mT5 subword predictions (where 1 token ≈ 3–6 characters) is severely insufficient for byte-level generation.

### Can We Rerun?
**ByT5 checkpoints are NOT available locally** (they were trained on Colab and not committed to the repository). Only mT5 checkpoints are available locally (`outputs/checkpoints/baseline_mt5/`, `outputs/checkpoints/multitask_mt5/`).

**mT5 could be re-evaluated** with a corrected generation limit, but since mT5 was already producing predictions with max length 87 (well above 16), the impact on mT5 metrics would be minimal.

**ByT5 cannot be re-evaluated** without its checkpoints. The results are systematically biased downward by an unknown amount.

### Severity: **BLOCKER**

### Finding
- All ByT5 EM and F1 metrics are artificially underestimated relative to mT5 due to the byte-level generation cap.
- The 16-byte cap affects ~23% of matchedqa_byt5 QA predictions directly.
- The truncation examples in the manuscript ("Thomas Alva Edis", "James Francis Ca", "Babagana Umara Z") are all exactly 16 bytes — confirming hard truncation.
- The claim that Matched-Vol ByT5 (EM=0.219) outperforms mT5 (EM=0.197) may or may not hold at a corrected limit.
- The claim that Multitask ByT5 (EM=0.192) underperforms Matched-Vol ByT5 (EM=0.219) may be confounded by the fact that Multitask ByT5 had 15.4% of QA predictions truncated vs 10.1% for Matched-Vol ByT5 (multitask had more NER examples which are generally longer — so the QA-only subset had even higher truncation proportionally).

### Action Required
1. Report all ByT5 metrics as *potentially underestimated* due to 16-byte generation cap.
2. Add a prominent method note stating the generation limit and its implication.
3. Retrain or re-evaluate on Colab at generation limits of 32, 64, 128 bytes to determine the true metrics. This is a **required experiment before submission**.
4. Update all tables with "(†generation cap 16 bytes)" annotation until corrected values are available.
5. Revise claims about ByT5 superiority and the task delta to be conditional on this caveat.

---

## Issue 2: Max Source Length Contradiction — MAJOR

### Evidence Inspected
- `configs/baseline_byt5_1x.yaml`: `model.max_source_length = 192` (unused inference field), `train.max_source_length = 256`
- `configs/matchedqa_byt5.yaml`: same values
- **Actual runtime logs (authoritative):**
  - `02_train_baseline_byt5_v2.log`: `Tokenization: max_source_length=1024, max_target_length=128`
  - `02_train_matchedqa_byt5.log`: `Tokenization: max_source_length=1024, max_target_length=128`
  - `03_train_multitask_qa.log`: Final ByT5 run: `Tokenization: max_source_length=1024, max_target_length=128`
  - `02_train_baseline_qa.log`: mT5 all runs: `max_source_length=256`
  - `02_train_matchedqa_mt5.log`: `max_source_length=256`

### Finding
The YAML configs have `train.max_source_length = 256` for all models, but the **runtime always used 1024 for ByT5** because the training scripts override `train.max_source_length` with a ByT5-specific value of 1024. The `model.max_source_length = 192` in the YAML is a legacy field not used by the training script.

The manuscript previously stated "mT5 max_source_length = 256; ByT5 max_source_length = 1024" which is **correct** for the actual runtime but was labelled as coming from the YAML (which shows 256). The YAML shows `train.max_source_length = 256`, but the actual runtime for ByT5 used 1024.

The source length stated in the manuscript (ByT5=1024, mT5=256) is **factually correct** per the logs. The contradiction was in the appendix YAML excerpt, which showed a misleading `train.max_source_length = 256` that was overridden at runtime.

### Note on multitask ByT5 training: The log shows some early failed runs used 256, and some runs used 1024. The **final successful run** (which produced the committed predictions) used 1024.

### Severity: **MAJOR** (but resolved — the manuscript values are correct, but must be sourced from logs not YAML)

### Action Required
1. Update the appendix YAML excerpt to remove the `train.max_source_length = 256` line (or annotate that it is overridden at runtime).
2. Add a footnote: "Effective runtime max_source_length values were confirmed from training logs; the YAML config value was overridden by the training script for ByT5."
3. Correct Table 4 (Hyperparameters) to add a note explaining the discrepancy.

---

## Issue 3: Learning Rate Asymmetry — MODERATE (Partially Resolved)

### Evidence Inspected
- `02_train_baseline_qa.log`: Run 1 at `lr=5e-4` → OOM/abandoned, Run 2 at `lr=1e-4` → checkpoint saved
- `02_train_matchedqa_mt5.log`: `lr=1e-4`
- `02_train_matchedqa_byt5.log`: `lr=1e-4`
- `02_train_baseline_byt5_v2.log`: `lr=1e-4`
- `03_train_multitask_qa.log`: All runs at `lr=1e-4`

### Finding
**All final experiments for both architectures used lr = 1×10⁻⁴.** The initial mT5 baseline run at 5×10⁻⁴ was abandoned without saving a checkpoint. The saved model, evaluated predictions, and all reported metrics used lr=1e-4 for both mT5 and ByT5.

**The manuscript's statement that "mT5 baseline trained at 5e-4 while ByT5 trained at 1e-4" is incorrect.** Both final architectures used 1e-4.

**There is no learning rate asymmetry to disclose.** The manuscript's explanation ("longer byte sequences increase gradient magnitudes") is both unsupported by evidence and factually wrong about the experiment.

### Severity: **MAJOR** — the manuscript contains a factually incorrect statement about learning rates.

### Action Required
1. Remove ALL references to the learning rate asymmetry between architectures.
2. State: "All experiments used lr = 1×10⁻⁴ (Adafactor). An initial mT5 baseline run at lr=5×10⁻⁴ was abandoned due to instability before checkpoint saving; it is not included in the reported results."
3. Remove the "methodological note on learning rates" from Table 4.
4. Update Limitations section to remove the learning rate confound claim.

---

## Issue 4: "Consistent" NER Harm Across Languages — MODERATE

### Evidence (from `outputs/analysis/all_tables.json`)
ByT5 task delta (EM metric) per language:
- Overall: −0.027
- Swahili: **+0.010** (POSITIVE)
- Hausa: −0.050
- Yoruba: −0.043

ByT5 task delta (F1 metric) per language:
- Overall: −0.020
- Swahili: **+0.013** (POSITIVE)
- Hausa: −0.041
- Yoruba: −0.033

ByT5 task delta (Semantic metric) per language:
- Overall: −0.037
- Swahili: **−0.030** (negative)
- Hausa: −0.044
- Yoruba: −0.036

### Count of negative ByT5 task deltas (of 9 language-metric combinations):
- EM: 2/3 negative (Swahili is positive)
- F1: 2/3 negative (Swahili is positive)
- Semantic: 3/3 negative
- **Total: 7 of 9 language-metric combinations show negative ByT5 task delta.**
- **2 of 9 (Swahili EM, Swahili F1) show positive task delta.**

### Severity: **MODERATE** — manuscript currently says "consistently across all languages and all metrics" which is false.

### Action Required
1. Change Abstract claim from "consistently degrades" to "degrades overall and in seven of nine language-metric comparisons"
2. Add "(the exception being Swahili EM and F1, where a small positive delta of +0.010 EM and +0.013 F1 was observed)"
3. Update Introduction, Contributions, Results §5.2, Discussion §7.2, and Conclusion similarly.
4. Note: The Swahili exception may reflect noise at single seed or a genuine language-specific effect; do not speculate further without evidence.

---

## Issue 5: "Matched-Volume" / "Step Parity" Claim — MAJOR

### Evidence Inspected
Training configurations confirmed from logs:
- matchedqa_byt5: 9,400 examples, eff_batch=64 → 147 steps/epoch
- multitask_byt5: 28,585 examples (9,400 QA + 19,185 NER), eff_batch=64 → 447 steps/epoch

Even if both ran for exactly the same number of epochs, the multitask experiment takes **3.04× more optimizer steps per epoch**. Additionally:
- matchedqa sees 9,400 QA examples per epoch; multitask sees only 9,400 QA examples per epoch (same), but also 19,185 NER examples.
- "Step parity" was an intended design, not a verified outcome.
- The logs do not record total steps completed at early stopping, so we cannot verify if they stopped at the same epoch.

### Finding
The matched-volume and multitask experiments are **not step-matched** in optimizer steps per epoch. The matched-volume experiment ran fewer optimizer steps per epoch. The claim of "step parity" in the manuscript is **not supported by the training design** as implemented.

What IS true: Both experiments use the same QA upsampling factor (20×), so both see 9,400 QA training examples per epoch. The difference is that multitask adds 19,185 NER examples per epoch.

### Severity: **MAJOR** — the central analytical framing ("step parity controls for exposure") is not technically accurate.

### Action Required
1. Remove all uses of "step parity" and "matched-volume" from paper.
2. Replace with: "QA-matched training" — both experiments see the same 9,400 QA examples per epoch, with the multitask condition additionally seeing NER examples.
3. Acknowledge that Δ_task is the marginal effect of NER supervision *given equal QA exposure per epoch*, not given equal optimizer steps.
4. Add Limitations entry noting that total optimizer step counts were not equated.
5. Rename experiment tables consistently: "QA-Upsampled (20×)" not "Matched-Vol".

---

## Issue 6: Translation Pipeline Description — MAJOR

### Evidence Inspected
- `scripts/02b_eval_translation_pipeline.py`: Full implementation visible
- Translation model: `facebook/nllb-200-distilled-600M`
- QA model: `google/flan-t5-base`
- Direction: Native language → English (NLLB) → flan-t5-base QA → English → Native language (NLLB)
- Context: The script translates **only the question**, not the passage. The `answer_english` function uses the translated question directly (no passage context provided to flan-t5).
- `02b_eval_translation_pipeline.log`: Confirms NLLB-200-distilled-600M + flan-t5-base used on A100 GPU

### Finding
The translation pipeline is:
1. Strip "question: " prefix from input
2. Translate question to English using **NLLB-200-distilled-600M**
3. Answer the English question using **FLAN-T5-base** (WITHOUT the passage context)
4. Translate the English answer back using **NLLB-200-distilled-600M**

This is **NOT a standard RAG or extractive QA pipeline.** The pipeline passes only the question to flan-t5-base, not the gold passage. This means flan-t5-base is doing **closed-book QA** (relying on parametric memory), not reading comprehension. This is a **severe methodological limitation** that must be disclosed prominently and makes the comparison less valid than described.

**The manuscript was incorrect in calling this "standard industrial practice."**

### Severity: **BLOCKER** — the translation pipeline baseline uses a non-reading-comprehension design (closed-book QA without passage context). This fundamentally changes the interpretation of the comparison.

### Action Required
1. Describe the pipeline precisely: NLLB-200-distilled-600M translation + FLAN-T5-base closed-book QA (question only, no passage).
2. Acknowledge that the translation pipeline is **not a fair QA baseline** because it discards the gold passage. The fine-tuned models read the passage; the translation pipeline does not.
3. Reframe the translation pipeline as "a translation-only parametric baseline" rather than a reading-comprehension comparison.
4. This substantially changes the interpretation: our fine-tuned models have an inherent advantage (they see the passage), but the comparison still demonstrates that even this disadvantaged translation baseline performs better than under-exposed direct QA models. The claim that direct QA (with proper exposure) beats it is still valid but for a different reason.
5. Remove "standard industrial practice" language.

---

## Issue 7: NER Parse Rate Causality — MODERATE

### Evidence
The manuscript states the 0.7% parse rate "is evidence of structural generation failure" and uses it as an "explanatory mechanism." This goes beyond what the data shows.

### Finding
The 0.7% parse rate is consistent with structural generation failure under this specific NER formulation, but it is not causal evidence for QA degradation. The manuscript draft used appropriate hedging in some places but not all (e.g., "providing an explanatory mechanism," "provides direct evidence").

### Severity: **MODERATE**

### Action Required
1. Replace "provides direct evidence that byte-level models largely fail at structural NER generation" → "is consistent with structural generation failure under this NER formulation"
2. Replace "providing an explanatory mechanism" → "suggesting a plausible mechanism"
3. Retain the parse rate observation as correlational, not causal.

---

## Issue 8: Dataset Counts and Reproducibility — MODERATE

### Evidence from `data/processed/qa_seq2seq_swa_hau_yor/`
Actual per-language split counts:

| Split | Hausa | Yoruba | Swahili | Total |
|---|---:|---:|---:|---:|
| Train | 223 | 247 | 0 | 470 |
| Validation | 222 | 0 | 0 | 222 |
| Test | 300 | 253 | 295 | 848 |

**Swahili is absent from train and validation splits.** This is a major finding: the ByT5 and mT5 models were never fine-tuned on Swahili QA examples. Swahili QA performance is entirely zero-shot for all fine-tuned models.

### Other Reproducibility Findings
- Repository URL: https://github.com/OmondiKevin/afriqa-entity-aware-qa (exists, but public/private status must be confirmed by user)
- Predictions: committed ✓
- Metrics: committed ✓
- Training logs: committed ✓
- Model checkpoints: mT5 checkpoints committed locally; ByT5 checkpoints NOT available

### Severity: **BLOCKER** — Swahili absence from training is an unreported finding that critically impacts interpretation. Per-language tables omitting Swahili from training make cross-language comparison potentially misleading.

### Action Required
1. Report per-language data splits including the Swahili absence from training.
2. Add note that all Swahili results represent **zero-shot generalization** (no Swahili QA examples in training). This may explain some performance patterns.
3. The Swahili positive ByT5 task delta (+0.010 EM) may reflect zero-shot NER transfer rather than a meaningful multitask effect.
4. Confirm repository visibility before claiming it is "publicly available."

---

## Issue 9: Bibliography — MODERATE

### Verified Corrections Needed

| Entry | Current | Correct |
|---|---|---|
| `ogundepo2023afriqa` | booktitle: Findings EMNLP 2023 | Correct — also add pages: 14957--14972 |
| `adelani2022masakhaner2` | booktitle: EMNLP 2022 | Correct — add pages: 4488--4508 |
| `nakkiran2019deepdoubledescent` | year: 2019, journal: arXiv | Should be venue: ICLR 2020 (published there, not just arXiv) |
| `artetxe2020xquad` | title: "On the Cross-lingual Transferability..." | Correct — but note: XQuAD is a dataset, not the main claim of the paper; add pages: 4623--4637 |
| `clark2020tydiqa` | booktitle: TACL | Should be @article, journal: TACL, volume: 8, pages: 454--470 |
| `longpre2021mkqa` | booktitle: TACL | Should be @article, journal: TACL, volume: 9, pages: 1389--1406 |
| `hu2022lora` | booktitle: ICLR | Correct |
| All ByT5, mT5, LaBSE entries | — | Appear correct |

### Relevance Assessment
- **The Bitter Lesson (Sutton 2019):** Weakly relevant — cited as motivational frame. Keep if mentioned in text, but do not lead with it.
- **Deep Double Descent (Nakkiran 2020):** Weakly relevant — mentioned speculatively. Either strengthen the connection or remove.
- **Caruana 1997 / Collobert 2011:** Standard multitask learning citations. Keep as background but are not tightly relevant to the specific findings.

### Severity: **MODERATE**

---

## Issue 10: Title Scope — MODERATE

### Finding
The current title "Scaling Exposure to Replace Translation Pipelines" overstates the findings given:
- Single seed
- Three languages
- One translation baseline (closed-book QA, not true reading comprehension pipeline)
- The translation pipeline was not a proper reading comprehension comparison (Issue 6)

With Issue 6 now documented (the translation pipeline doesn't read the passage), the title "to Replace Translation Pipelines" is even weaker — we're beating a closed-book parametric baseline, not a true extraction pipeline.

### Recommended Title
**"Exposure Scaling for Direct Multilingual Question Answering in Low-Resource African Languages"**

This is more accurate, describes what was actually done, and does not overclaim.

### Severity: **MODERATE**

### Action Required
1. Change title to the above or a close variant.
2. Review all "replace translation pipelines" language in body text.
3. Ensure abstract and conclusion do not claim translation pipelines are superseded.

