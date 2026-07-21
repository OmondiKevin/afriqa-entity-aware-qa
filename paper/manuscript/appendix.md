# Appendix

---

## A. Hyperparameter Details

**Table A1: Training configuration per experiment.** All values confirmed from training logs. Discrepancies between YAML config values and runtime values are noted.

| Parameter | mT5 Baseline | ByT5 Baseline | mT5 QA-Ups. | ByT5 QA-Ups. | mT5 Multi. | ByT5 Multi. |
|---|---|---|---|---|---|---|
| Model | google/mt5-base | google/byt5-base | mt5-base | byt5-base | mt5-base | byt5-base |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Optimizer | Adafactor | Adafactor | Adafactor | Adafactor | Adafactor | Adafactor |
| Scheduler | Linear | Linear | Linear | Linear | Linear | Linear |
| Physical Batch | 64 | 32 | 64 | 64 | 64 | 64 |
| Grad. Accum. | 1 | 2 | 1 | 1 | 1 | 1 |
| Effective Batch | 64 | 64 | 64 | 64 | 64 | 64 |
| max_source_len | 256 (tokens) | 1024 (bytes)* | 256 | 1024* | 256 | 1024* |
| max_target_len | 128 (tokens) | 128 (bytes)** | 128 | 128** | 128 | 128 |
| gen_max_new_tok | 16 (tokens) | **16 (bytes)†** | 16 | **16†** | 16 | **16†** |
| Epochs (max) | 20 | 20 | 20 | 20 | 20 | 20 |
| Early Stop. | 3 | 3 | 3 | 3 | 3 | 3 |
| Warmup | 50 | 50 | 50 | 50 | 50 | 50 |
| Precision | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 |
| Seed | 42 | 42 | 42 | 42 | 42 | 42 |
| QA Upsample | 1× | 1× | 20× | 20× | 20× | 20× |

\* The YAML config `train.max_source_length = 256` was overridden at runtime to 1024 for all ByT5 experiments. Confirmed from training logs.  
\*\* max_target_length = 128 bytes sets the training truncation boundary. Evaluation used 16 bytes (`gen_max_new_tokens`).  
† **Critical:** 16 bytes ≈ 13–16 ASCII characters. 24.1% of gold answers exceed 16 bytes. All ByT5 QA metrics are lower bounds.

---

## B. Dataset Statistics

**Table B1: AfriQA gold-passage QA dataset, per-language split counts.**

| Language | Train | Validation | Test |
|---|---:|---:|---:|
| Hausa | 223 | 222 | 300 |
| Yoruba | 247 | 0 | 253 |
| Swahili | **0** | **0** | 295 |
| **Total** | **470** | **222** | **848** |

> **Zero-shot Swahili:** Swahili is entirely absent from training and validation splits. All Swahili performance figures in this paper represent zero-shot cross-lingual generalization from Hausa and Yoruba training data. This applies equally to all fine-tuned model configurations.

**Table B2: MasakhaNER 2.0 dataset, total examples (Hausa + Yoruba + Swahili combined).**

| Split | Examples |
|---|---:|
| Train | 19,185 |
| Validation | 2,741 |
| Test | 5,480 |
| **Total** | **27,406** |

---

## C. Per-Language Metrics (Extended)

**Table C1: Per-language EM, F1, and semantic similarity for all configurations.** SWA = Swahili (zero-shot); HAU = Hausa; YOR = Yoruba. †Lower bound (16-byte generation cap).

| Configuration | SWA EM | SWA F1 | SWA Sem | HAU EM | HAU F1 | HAU Sem | YOR EM | YOR F1 | YOR Sem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline mT5 | 0.017 | 0.055 | 0.312 | 0.030 | 0.057 | 0.273 | 0.051 | 0.105 | 0.292 |
| Baseline ByT5† | 0.003 | 0.058 | 0.327 | 0.017 | 0.046 | 0.318 | 0.059 | 0.122 | 0.340 |
| QA-Ups. mT5 | 0.176 | 0.214 | 0.419 | 0.227 | 0.261 | 0.453 | 0.186 | 0.233 | 0.447 |
| QA-Ups. ByT5† | 0.180 | 0.253 | 0.499 | 0.277 | 0.334 | 0.541 | 0.198 | 0.287 | 0.502 |
| Multitask mT5 | 0.176 | 0.224 | 0.443 | 0.277 | 0.306 | 0.498 | 0.170 | 0.255 | 0.465 |
| Multitask ByT5† | 0.190 | 0.266 | 0.469 | 0.227 | 0.293 | 0.497 | 0.154 | 0.254 | 0.464 |
| LoRA mT5 | 0.142 | 0.174 | 0.384 | 0.197 | 0.235 | 0.415 | 0.142 | 0.192 | 0.397 |
| LoRA ByT5† | 0.034 | 0.080 | 0.305 | 0.057 | 0.113 | 0.318 | 0.040 | 0.090 | 0.301 |
| Translation | 0.122 | 0.202 | — | 0.157 | 0.234 | — | 0.059 | 0.122 | — |

**Table C2: ByT5 task delta (Multitask ByT5 − QA-Upsampled ByT5) per language and metric.** Positive = NER supervision helps; Negative = NER supervision hurts.

| Language | EM Δ | F1 Δ | Sem Δ |
|---|---:|---:|---:|
| Swahili (zero-shot) | **+0.010** | **+0.013** | −0.030 |
| Hausa | −0.050 | −0.041 | −0.044 |
| Yoruba | −0.043 | −0.033 | −0.038 |
| **Overall** | **−0.027** | **−0.020** | **−0.037** |

Negative in 7 of 9 language-metric combinations. Positive only for Swahili EM and Swahili F1, where the zero-shot condition makes interpretation unreliable.

---

## D. Training Step Analysis

**Table D1: QA exposure and optimizer steps per configuration epoch.** This table documents why Δ_task reflects equal-QA-exposure-per-epoch, not equal optimizer steps.

| Configuration | QA examples/epoch | NER examples/epoch | Total examples/epoch | Optimizer steps/epoch |
|---|---:|---:|---:|---:|
| Baseline (1×) | 470 | 0 | 470 | 8 |
| QA-Upsampled (20×) | 9,400 | 0 | 9,400 | 147 |
| Multitask (20× QA) | 9,400 | 19,185 | 28,585 | 447 |

*Optimizer steps = ceil(examples / effective_batch_size = 64)*

The Multitask configuration takes **3.04× more optimizer steps per epoch** than the QA-Upsampled configuration. Because Δ_task = Multitask − QA-Upsampled, the task delta conflates the effect of NER supervision with 3.04× additional compute per epoch. This is a limitation of the current experimental design.

---

## E. NER Evaluation Details

**Table E1: NER generation analysis on MasakhaNER 2.0 test set.**

| Metric | Multitask mT5 | Multitask ByT5 |
|---|---:|---:|
| Total predictions | 5,480 | 5,480 |
| Parseable predictions | 409 (7.5%) | 41 (0.7%) |
| Precision (on parseable) | 0.840 | 0.122 |
| Recall (over all examples) | 0.036 | ~0.000 |
| F1 (token-level) | 0.070 | 0.001 |

"Parseable" means the prediction matched the expected tag-delimited format (e.g., "PER: Odunayo Ogundepo, LOC: Nigeria"). The 0.7% parse rate for ByT5 indicates the model largely did not acquire the NER output format, making its NER training signal functionally unreliable for the vast majority of examples.

---

## F. Translation Pipeline Details

The translation baseline was implemented entirely using open-source neural models, with no commercial API calls:

**Stage 1:** NLLB-200-distilled-600M [NLLB Team, 2022] translates the source-language *question string* (passage not included) into English. The prompt prefix "question: " is stripped before translation.

**Stage 2:** FLAN-T5-base generates an English answer from the translated English question only. **The gold passage is not provided.** This makes the pipeline a *closed-book parametric baseline* relying on world-knowledge in FLAN-T5-base parameters.

**Stage 3:** NLLB-200-distilled-600M translates the English answer back into the source language.

**Implications:** The fine-tuned direct models (Configurations 1, 3, 4, 5) read the gold passage during inference. The translation baseline does not. This informational asymmetry advantages direct models. The comparison demonstrates that even with passage access, severely under-exposed (1×) models fail to outperform the closed-book translation pipeline; 20× exposure resolves this. It does not demonstrate superiority over a passage-reading translation system.

---

## G. Qualitative Error Examples

**Table G1: Characteristic error patterns with corrected generation-cap annotation.**

**Category 1: Byte-level truncation (artifact of 16-byte cap)**

| Model | Gold | Prediction | Analysis |
|---|---|---|---|
| ByT5 QA-Ups. | Thomas Alva Edison (18 bytes) | Thomas Alva Edis (16 bytes) | Hard truncation |
| ByT5 QA-Ups. | James Francis Cameron (21 bytes) | James Francis Ca (16 bytes) | Hard truncation |
| ByT5 QA-Ups. | Babagana Umara Zulum (21 bytes) | Babagana Umara Z (16 bytes) | Hard truncation |
| ByT5 QA-Ups. | Adam Wilhelm Moltke (20 bytes) | Adam Wilhelm Mol (16 bytes) | Hard truncation |

*These predictions would likely score EM=1 at a 128-byte generation limit. They currently score F1 ≈ 0.67.*

**Category 2: Semantic domain hallucination**

| Model | Gold | Prediction | Analysis |
|---|---|---|---|
| mT5 QA-Ups. | Doha | Al Jazeera | Correct domain (Qatar/media), wrong entity |
| ByT5 QA-Ups. | Mt Petro | Annuario Pontifi[cio] | Vatican domain preserved, wrong referent |
| Multitask mT5 | Webuye | 1992 | Entity type confusion (name→year) |

**Category 3: Translation-pipeline hallucination**

| Gold | NLLB-translated Q | FLAN-T5 Answer | Back-translated | Analysis |
|---|---|---|---|---|
| Oginga Odinga | Who founded FORD-Kenya? | Oginga Odinga | Jaramogi Oginga Odinga | Correct but expanded (back-translation adds "Jaramogi") |
| Ama Ata Aidoo | Who wrote "Our Sister Killjoy"? | Ama Ata Aidoo | Ama Ata Aidoo | Correct (FLAN-T5 world knowledge) |
| 2021 | In what year did a woman first become US VP? | 2021 | 2021 | Correct |

The translation baseline shows strong performance on high-frequency world-knowledge facts and poor performance on African-specific facts requiring local passage knowledge.

**Category 4: NER format leakage into QA (mT5 Multitask)**

| Input | Gold | Prediction | Analysis |
|---|---|---|---|
| question: Who discovered X? | John Smith | PER: John Smith | NER format prefix leaked into QA |

This leakage pattern (1.8% of mT5 Multitask predictions) does not occur in ByT5 Multitask (0.4%), consistent with ByT5's low NER format acquisition rate.

