# Exposure Scaling for Direct Multilingual Question Answering in Low-Resource African Languages

**Authors:** [Author Name(s)]  
**Affiliation:** [Institution(s)]  
**Correspondence:** [email@institution.edu]

---

## Abstract

Cross-lingual question answering (QA) systems for African languages have historically depended on pipelines that translate source-language input into English for downstream inference, compounding errors across morphologically diverse language families. In this work, we investigate whether direct multilingual QA — where models read and answer questions entirely in the native language — can match or exceed these pipelines, and we identify **gradient exposure volume** as the central variable governing that transition.

We conduct a controlled ablation study across two model architectures (mT5-base and ByT5-base), five training configurations, and three African languages (Swahili, Hausa, and Yoruba) using the AfriQA benchmark supplemented with MasakhaNER 2.0 supervision. Our primary finding is that a translation-based parametric baseline (EM = 0.116) is outperformed by a byte-level model trained with 20× QA upsampling (EM = 0.219†, F1 = 0.292†) — an improvement achieved without cross-lingual heuristics, simply by ensuring sufficient gradient update steps. We note that these ByT5 figures are generated under a constrained decoding limit (16 bytes) and represent a lower bound; corrected values are pending re-evaluation (see §4.4 and §8).

A secondary finding concerns the interaction between tokenization architecture and multitask Named Entity Recognition (NER) supervision. For subword models (mT5), NER supervision on top of QA-upsampled training yields a positive delta (EM +0.013; F1 +0.026). For byte-level models (ByT5), the same supervision yields a negative overall delta (EM −0.027; F1 −0.020) across two of three languages. The exception is Swahili (EM +0.010; F1 +0.013 for ByT5 Multitask), a finding we attribute to the fact that Swahili was absent from training data and all Swahili results reflect zero-shot generalization.

All reported results are based on a single random seed and must be treated as empirically motivated hypotheses pending multi-seed validation and corrected ByT5 re-evaluation.

**†** Lower-bound estimate; generation constrained to 16 bytes. See §4.4.

**Keywords:** low-resource QA, African NLP, byte-level models, multilingual learning, multitask learning, LoRA, AfriQA, MasakhaNER

---

## 1. Introduction

Question answering in low-resource African languages faces a persistent infrastructure gap: small training corpora, morphologically complex language families, and historically English-centric NLP tooling. The practical response has been to route all inference through English: translate the source question into English, apply an English QA model, and translate the answer back. This approach introduces at least two points of compounding error and discards the passage entirely in simpler implementations.

Direct, native-language QA sidesteps these failure modes but faces a practical barrier: with only a few hundred training examples in a given African language, naively fine-tuned sequence-to-sequence models perform poorly. Prior work has typically interpreted this failure as evidence of architectural or data insufficiency, motivating increasingly complex multi-task or cross-lingual transfer schemes.

This paper challenges that interpretation. We propose that the primary bottleneck is **gradient exposure volume**: the number of parameter updates the model receives on the target task. Because African QA datasets are small (470 examples across three languages in our setup), a model trained for the conventional number of epochs receives far fewer gradient updates than one trained on a multitask corpus. Once this exposure gap is closed — by upsampling the QA data 20× to achieve epoch-level parity with the multitask setup — direct native-language models consistently exceed the translation-based baseline.

We also investigate whether the utility of explicit NER supervision depends on tokenization architecture, and find evidence for an interaction effect — though with the important caveat that all results come from a single seed and one experiment configuration, and that a decoding constraint artificially suppresses ByT5 metrics throughout (see §4.4).

Our contributions are:

1. **Evidence that gradient exposure volume dominates over NER supervision** in determining whether direct QA models match translation-based baselines, in our three-language setting.

2. **Evidence for a tokenization-supervision interaction** (positive for mT5; negative overall for ByT5), with caveats for single-seed experiments, Swahili zero-shot conditions, and constrained ByT5 decoding.

3. **Documentation of a methodological issue** (16-byte generation cap for ByT5) that systematically under-estimates ByT5 metrics relative to mT5; we provide a plan for corrected re-evaluation.

4. **A publicly released evaluation framework** covering nine configurations, three languages, and three metrics on the AfriQA test set, with all predictions, metrics, and logs version-controlled.

---

## 2. Related Work

### 2.1 Multilingual and Cross-Lingual QA

Extractive and generative QA has been studied in multilingual settings through benchmarks including XQuAD [Artetxe et al., 2020], TyDi QA [Clark et al., 2020], and MKQA [Longpre et al., 2021]. These benchmarks primarily cover high- and mid-resource languages. AfriQA [Ogundepo et al., 2023] fills this gap with gold-passage reading comprehension tasks in twelve African languages, including the three studied here. The NLLB team [NLLB Team, 2022] demonstrates the scale of compute required for reasonable NMT in low-resource languages, contextualizing why translation remains a practical baseline despite its known limitations.

### 2.2 Multilingual Pretrained Models

mT5 [Xue et al., 2021] extends T5 to 101 languages with a SentencePiece vocabulary. ByT5 [Xue et al., 2022] processes raw UTF-8 byte sequences, trading tokenization complexity for longer input representations. For passage-level QA, byte-level encoding typically yields 3–4× longer sequences than subword encoding of the same text.

### 2.3 Multitask and Entity-Aware Learning

Multitask learning as a regularizer has a long history in NLP [Caruana, 1997; Collobert et al., 2011]. MasakhaNER 2.0 [Adelani et al., 2022] provides Africa-focused NER annotations for PER, LOC, ORG, and DATE categories across 20+ languages, including our three target languages. The interaction between multitask NER supervision and tokenization architecture has not been systematically studied for African languages.

### 2.4 Parameter-Efficient Fine-Tuning

LoRA [Hu et al., 2022] inserts low-rank decomposition matrices into frozen pretrained weights. Its effectiveness under multitask conditions with byte-level architectures has not been previously assessed.

### 2.5 Exposure and Generalization

Sutton's Bitter Lesson [Sutton, 2019] argues that general methods leveraging computation tend to dominate human-engineered inductive biases. Our finding that scaled QA exposure outperforms NER-augmented training for ByT5 is consistent with this view, though we refrain from strong causal claims given single-seed constraints. Deep Double Descent [Nakkiran et al., 2020] is noted as background for the non-trivial relationship between training volume and generalization.

---

## 3. Methodology

### 3.1 Problem Formulation

Given a question *q* and a gold passage *p* in a native African language *l*, a model generates an answer string *â*, evaluated against a gold answer *a*. We use gold passages from AfriQA and do not study retrieval.

### 3.2 Experimental Configurations

We design five configurations varying gradient exposure volume, tokenization architecture, and NER supervision.

**Configuration 1 — Low-Exposure Baseline (1×):**  
Both mT5-base and ByT5-base are fine-tuned on the raw AfriQA training split (470 examples, Hausa + Yoruba only; Swahili is absent from training — see §4.1) for 20 epochs. With effective batch size 64, this yields approximately 147 gradient steps.

**Configuration 2 — Translation-Based Parametric Baseline:**  
Source-language questions are translated into English using NLLB-200-distilled-600M [NLLB Team, 2022]. English answers are generated using FLAN-T5-base [CITATION NEEDED: Chung et al. 2022] as a closed-book QA system (question only; passage not provided). English answers are translated back to the source language using NLLB-200-distilled-600M.

> **Important limitation:** Because FLAN-T5-base receives only the translated question and not the gold passage, this pipeline performs closed-book QA from parametric memory, not reading comprehension. The fine-tuned direct models have access to the passage and therefore have an inherent informational advantage. The comparison demonstrates that even this disadvantaged translation approach outperforms severely under-exposed direct QA, and that properly exposed direct QA can exceed it. It does not demonstrate that direct QA would beat a proper passage-reading translation pipeline.

**Configuration 3 — QA-Upsampled (20×):**  
AfriQA training examples are upsampled 20× (9,400 effective examples) and training runs for up to 200 epochs with early stopping (patience = 3). This ensures the model sees the same 9,400 QA examples per epoch as the multitask configuration. No NER data is included.

**Configuration 4 — Multitask NER + QA:**  
The 20× upsampled QA examples (9,400) are interleaved with MasakhaNER 2.0 training examples (19,185 NER instances), giving 28,585 total examples per epoch — 3.04× more optimizer steps per epoch than Configuration 3. NER examples use prompt prefix "extract entities:"; QA examples use "question:".

**Configuration 5 — LoRA Multitask:**  
Identical to Configuration 4 with rank-16 LoRA adapters (α = 32, dropout = 0.05) applied to attention projections; adapter parameters only are trained.

### 3.3 Exposure Delta Decomposition

We define:

- **Δ_exposure** = Performance(QA-Upsampled 20×) − Performance(Baseline 1×)
- **Δ_task** = Performance(Multitask) − Performance(QA-Upsampled 20×)
- **Δ_total** = Δ_exposure + Δ_task

**Important caveat on "step parity":** Configurations 3 and 4 are matched on *QA examples per epoch* (both see 9,400 QA examples/epoch), but Configuration 4 takes 3.04× more optimizer steps per epoch due to NER examples. Δ_task therefore reflects the marginal effect of NER supervision given equal QA exposure per epoch, not given equal total optimizer steps. This distinction is noted in the Limitations.

---

## 4. Experimental Setup

### 4.1 Datasets

**AfriQA (gold-passage subset)** [Ogundepo et al., 2023]:

| Split | Hausa | Yoruba | Swahili | Total |
|-------|------:|-------:|--------:|------:|
| Train | 223 | 247 | **0** | 470 |
| Validation | 222 | 0 | **0** | 222 |
| Test | 300 | 253 | 295 | 848 |

> **Critical note:** Swahili is entirely absent from the training and validation splits. All Swahili results in this paper reflect **zero-shot generalization** — the models were never trained on Swahili QA examples. This applies to both mT5 and ByT5 across all configurations.

**MasakhaNER 2.0** [Adelani et al., 2022]:

| Split | Examples |
|-------|--------:|
| Train | 19,185 |
| Validation | 2,741 |
| Test | 5,480 |

### 4.2 Models

| Model | Parameters | Tokenization | Effective Max Source Length |
|-------|----------:|------------|---:|
| mT5-base | ~580M | SentencePiece subword | 256 tokens |
| ByT5-base | ~582M | UTF-8 byte sequences | 1,024 bytes |

Max source lengths confirmed from training logs (`Tokenization: max_source_length=256` for mT5; `max_source_length=1024` for ByT5). YAML config values were overridden by the training script for ByT5.

### 4.3 Evaluation Metrics

- **Exact Match (EM):** Fraction of normalized predictions matching the gold answer exactly.
- **Token F1:** Bag-of-words token F1 averaged over test examples.
- **Semantic Similarity:** Cosine similarity of LaBSE [Feng et al., 2022] sentence embeddings. Not reported for the translation baseline (back-translation artifacts).

### 4.4 Generation Limit and ByT5 Correction Caveat

All experiments used `generation_max_new_tokens = 16` in the evaluation configuration. For mT5 (subword), 16 tokens encodes approximately 60–100 characters — sufficient for nearly all gold answers in this dataset (observed maximum prediction length: 87 bytes). For ByT5 (byte-level), 16 bytes encodes only 13–16 ASCII characters, or fewer characters with multi-byte diacritics present in Hausa and Yoruba.

**Impact:** 24.1% of gold answers (n=205) exceed 16 bytes in length. For the QA-Upsampled ByT5 configuration, 10.1% of predictions (86/848) are demonstrably truncated (prediction exactly 16 bytes; gold answer longer). The truncation pattern is visible in qualitative examples (predictions such as "Thomas Alva Edis", "James Francis Ca").

**All ByT5 QA metrics reported in this paper are lower-bound estimates.** Corrected metrics at 128-byte generation limit are pending Colab re-evaluation of the ByT5 checkpoints (see §10 and the Evaluation Correction Report in the supplementary materials). mT5 metrics are unaffected by this issue.

Tables 1–3 annotate ByT5 values with **†** to indicate they are lower bounds under the 16-byte constraint.

### 4.5 Training Configuration

All final experiments used the following hyperparameters (confirmed from training logs):

| Hyperparameter | mT5-base | ByT5-base |
|---------------|---------|---------|
| Optimizer | Adafactor | Adafactor |
| Learning Rate | 1 × 10⁻⁴ | 1 × 10⁻⁴ |
| Effective Batch Size | 64 | 64 |
| Physical Batch / Grad Accum | 64 / 1 | 32 / 2 |
| Precision | bf16 | bf16 |
| Weight Decay | 0.01 | 0.01 |
| Max Gradient Norm | 1.0 | 1.0 |
| Warmup Steps | 50 | 50 |
| Early Stopping Patience | 3 | 3 |
| Random Seed | 42 | 42 |

> **Note on learning rates:** All final experiments for both architectures used lr = 1×10⁻⁴. An initial mT5 baseline run at lr = 5×10⁻⁴ was abandoned before checkpoint saving and is not included in the reported results.

### 4.6 Hardware

Single NVIDIA A100-SXM4-80GB GPU, 167 GB system RAM, Google Colab Pro. ByT5 checkpoints are not committed to the repository (due to size); mT5 checkpoints are available locally. All predictions, metrics, and training logs are version-controlled.

---

## 5. Results

### 5.1 Overall QA Performance

**Table 1: Overall QA performance on the AfriQA test set (n = 848).** Best per metric in **bold**. †All ByT5 values are lower-bound estimates due to 16-byte generation cap; see §4.4. —Semantic similarity not reported for translation baseline.

| Configuration | EM | F1 | Sem. |
|---|---:|---:|---:|
| Baseline mT5 (1×) | 0.032 | 0.071 | 0.293 |
| Baseline ByT5 (1×) | 0.025† | 0.073† | 0.327† |
| Translation Baseline (NLLB + FLAN-T5) | 0.116 | 0.189 | — |
| QA-Upsampled mT5 (20×) | 0.197 | 0.236 | 0.439 |
| QA-Upsampled ByT5 (20×) | 0.219† | 0.292† | 0.513† |
| Multitask mT5 (NER + QA) | **0.210** | 0.262 | 0.469 |
| Multitask ByT5 (NER + QA) | 0.192† | 0.272† | 0.476† |
| LoRA mT5 | 0.164 | 0.200 | 0.398 |
| LoRA ByT5 | 0.045† | 0.094† | 0.308† |

> Best EM (0.219†) is ByT5 QA-Upsampled, but this is a lower bound. Best verified EM is Multitask mT5 (0.210) or QA-Upsampled mT5 (0.197) for the mT5 family.

Key observations (subject to ByT5 correction caveat):

1. **Under 1× training, the translation baseline outperforms direct QA models.** mT5 EM = 0.032; ByT5 EM = 0.025† vs translation EM = 0.116.

2. **QA-upsampled direct models exceed the translation baseline.** QA-Upsampled ByT5 EM = 0.219† and mT5 EM = 0.197 both exceed 0.116. The comparison is clean for mT5; for ByT5 it holds even under the 16-byte constraint.

3. **NER supervision has opposing directional effects by architecture** (see §5.2). This finding holds qualitatively even under the 16-byte constraint but magnitudes may shift.

4. **LoRA collapses for ByT5.** LoRA ByT5 EM = 0.045† barely exceeds the 1× baseline (0.025†), while LoRA mT5 (EM = 0.164) retains 78% of Multitask mT5 performance.

### 5.2 Exposure Delta Decomposition

**Table 2: Decomposition of performance gains (EM, F1, Semantic Similarity).** Δ_exposure = QA-Upsampled − Baseline. Δ_task = Multitask − QA-Upsampled. †ByT5 values are lower bounds.

| Architecture | Metric | Baseline | QA-Upsamp. | Multitask | Δ_exposure | Δ_task |
|---|---|---:|---:|---:|---:|---:|
| mT5 | EM | 0.032 | 0.197 | 0.210 | +0.165 | +0.013 |
| mT5 | F1 | 0.071 | 0.236 | 0.262 | +0.165 | +0.026 |
| mT5 | Sem. | 0.293 | 0.439 | 0.469 | +0.147 | +0.030 |
| ByT5† | EM | 0.025 | 0.219 | 0.192 | +0.195 | −0.027 |
| ByT5† | F1 | 0.073 | 0.292 | 0.272 | +0.219 | −0.020 |
| ByT5† | Sem. | 0.327 | 0.513 | 0.476 | +0.186 | −0.037 |

The exposure delta dominates for both architectures (EM: +0.165 for mT5; +0.195† for ByT5). The task delta is directionally positive for mT5 (all three metrics) and directionally negative for ByT5 overall (all three metrics), with the exception of Swahili (see §5.3). These directional patterns hold qualitatively but magnitudes for ByT5 require correction.

### 5.3 Per-Language Results

**Table 3: Per-language results (EM / F1).** SWA = Swahili (zero-shot for all fine-tuned models); HAU = Hausa; YOR = Yoruba. †ByT5 values are lower bounds. Bold = best per language per metric.

| Configuration | SWA EM | SWA F1 | HAU EM | HAU F1 | YOR EM | YOR F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline mT5 (1×) | 0.017 | 0.055 | 0.030 | 0.057 | 0.051 | 0.105 |
| Baseline ByT5 (1×) | 0.003† | 0.058† | 0.017† | 0.046† | 0.059† | 0.122† |
| QA-Upsampled mT5 | 0.176 | 0.214 | 0.227 | 0.261 | 0.186 | 0.233 |
| QA-Upsampled ByT5 | 0.180† | 0.253† | **0.277†** | **0.334†** | 0.198† | **0.287†** |
| Multitask mT5 | 0.176 | 0.224 | **0.277** | 0.306 | 0.170 | 0.255 |
| Multitask ByT5 | **0.190†** | **0.266†** | 0.227† | 0.293† | 0.154† | 0.254† |
| Translation Baseline | 0.122 | 0.202 | 0.157 | 0.234 | 0.059 | 0.122 |

> Swahili ByT5 Multitask (EM = 0.190†, F1 = 0.266†) nominally exceeds QA-Upsampled ByT5 Swahili (EM = 0.180†, F1 = 0.253†), yielding a positive Swahili task delta. However, all Swahili results are zero-shot (no Swahili training examples), and this small positive difference at a single seed may reflect initialization noise rather than a genuine effect of NER supervision.

**Per-language ByT5 task deltas (EM):**
- Swahili: **+0.010†** (positive; but zero-shot; noise likely)
- Hausa: **−0.050†** (negative)
- Yoruba: **−0.043†** (negative)
- Overall: **−0.027†** (negative)

**Summary:** The aggregate ByT5 task delta is negative across all three overall metrics, and negative in seven of nine language-metric comparisons. The two exceptions (Swahili EM and F1) occur in the zero-shot language where noise dominates.

---

## 6. Error Analysis

**Table 4: Error distribution across all configurations (n = 848 for QA-only rows).**

| Configuration | EM (%) | Hi F1 >0.5 (%) | Lo F1 ≤0.5 (%) | Wrong F1=0 (%) | Empty (%) | NER Leak (%) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline mT5 (1×) | 3.2 | 2.7 | 6.0 | 87.9 | 0.2 | 0.0 |
| Baseline ByT5 (1×)† | 2.5 | 2.8 | 7.4 | 87.3 | 0.0 | 0.0 |
| Translation Baseline | 11.6 | 4.1 | 15.9 | 68.2 | 0.2 | 0.0 |
| QA-Upsampled mT5 | 19.7 | 3.7 | 3.4 | 73.2 | 0.0 | 0.0 |
| QA-Upsampled ByT5† | 21.9 | 6.4 | 7.0 | 64.7 | 0.0 | 0.0 |
| Multitask mT5 | 21.0 | 4.6 | 4.5 | 68.2 | 0.0 | 1.8 |
| Multitask ByT5† | 19.2 | 7.5 | 6.6 | 66.3 | 0.0 | 0.4 |
| LoRA mT5 | 16.4 | 3.4 | 3.2 | 77.0 | 0.0 | 0.0 |
| LoRA ByT5† | 4.5 | 5.0 | 3.9 | 86.7 | 0.0 | 0.0 |

†ByT5 high-F1 partial rates are inflated by the 16-byte truncation pattern (correct partial predictions that terminate at the byte cap). Correcting the generation limit will shift some "Hi F1" rows to "EM" and reduce the truncation artifact.

The translation baseline has an elevated low-F1 partial rate (15.9%), consistent with cascaded translation errors that partially preserve semantic content but distort surface form.

NER format leakage is present in Multitask mT5 (1.8%) and minimal in Multitask ByT5 (0.4%). This asymmetry is consistent with ByT5's failure to generate structured NER output (§6.1).

### 6.1 NER Evaluation

**Table 5: NER prediction evaluation (MasakhaNER test set, n = 5,480).**

| Model | Parseable Preds | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Multitask mT5 | 409 / 5,480 (7.5%) | 0.840 | 0.036 | 0.070 |
| Multitask ByT5 | 41 / 5,480 (0.7%) | 0.122 | ~0.000 | 0.001 |

ByT5 generates only 41 parseable NER annotations out of 5,480 test examples (0.7%), compared to 409 for mT5 (7.5%). Both parse rates are low; both models primarily fail to generate the expected tag-delimited format. The ByT5 failure is more severe. This observation is **consistent with** a structural generation failure under this NER formulation, and **may suggest** a plausible mechanism for the negative ByT5 task delta (shared training budget with an ineffective supervision signal). It does not establish a causal relationship.

### 6.2 Qualitative Error Patterns

**Byte-level truncation (artifact of generation cap).** QA-Upsampled ByT5 frequently predicts correct answers truncated at exactly 16 bytes: "Thomas Alva Edis" (gold: "Thomas Alva Edison"), "James Francis Ca" (gold: "James Francis Cameron"). These score F1 ≈ 0.67, inflating partial-match rates. Under a corrected generation limit, these would likely become exact matches.

**Semantic domain hallucination.** In the wrong-prediction category, several examples show plausible alternative referents: Gold: "Doha" → Pred: "Al Jazeera"; Gold: "Oluṣẹgun ọbasanjọ" → Pred: "General Yakubu G...". Models retrieve plausible world-knowledge entities from the correct domain but the wrong specific referent.

---

## 7. Discussion

### 7.1 Exposure Scaling

The dominant observation in Table 2 is the exposure delta: +0.165 EM for mT5 and +0.195† EM for ByT5. The mT5 figure is clean; the ByT5 figure is a lower bound. Even the mT5-only evidence is sufficient to establish that a single upsampling factor of 20× transforms a below-translation-baseline model into one that exceeds it. This result holds cleanly for mT5 without any generation-limit caveat.

We emphasize that the translation baseline used here (NLLB + FLAN-T5 closed-book) does not read the passage, giving fine-tuned direct models an informational advantage. Nevertheless, the under-exposed 1× models still underperformed it substantially (mT5 EM 0.032 vs 0.116), demonstrating that model access to the passage is insufficient without adequate gradient exposure. Proper exposure (20× upsampling) then closes and exceeds the gap.

### 7.2 Tokenization-Supervision Interaction

The mT5 task delta is positive across all three overall metrics. The ByT5 task delta is negative across all three overall metrics, and negative in seven of nine language-metric combinations. The two exceptions occur in Swahili, which had zero training examples — making it an unreliable basis for architectural claims.

We offer two non-mutually-exclusive candidate mechanisms (both remain speculative without probing experiments):

**Mechanism 1:** SentencePiece tokenization may fragment entity spans in agglutinative African languages, creating a fragmentation problem that explicit NER supervision corrects. Byte-level encoding preserves character boundaries exactly and does not exhibit this problem.

**Mechanism 2:** Byte-level generation of raw UTF-8 and structured tag-delimited NER output impose conflicting distributional pressures. The near-zero NER parse rate for ByT5 (0.7%) suggests the structural NER template was largely not acquired, leaving the QA task perturbed by a training signal that was largely non-functional.

These mechanisms are **consistent with** the data but are not established by it. Targeted ablations (single-task ByT5 trained on NER only; probing representations for entity boundary information) are needed for any causal claim.

### 7.3 LoRA Adapter Performance

LoRA ByT5 (EM = 0.045†) retains approximately 23% of Multitask ByT5 performance. LoRA mT5 (EM = 0.164) retains approximately 78%. The asymmetry suggests byte-level models require more distributed weight adaptation than LoRA rank-16 attention adapters provide — but this is speculative without ablation over adapter placement and rank.

---

## 8. Required Correction: ByT5 Re-Evaluation

The following experiment must be completed before this paper is submitted:

**Re-evaluate all ByT5 checkpoints at generation limits of 32, 64, and 128 bytes.** For each limit, report:
1. Overall EM, F1, semantic similarity
2. Per-language EM and F1
3. Error category distribution
4. Truncation rate (predictions exactly at limit with shorter gold)
5. Identify the limit at which metrics stabilize

The checkpoints required are:
- `baseline_byt5_v2` (ByT5 Baseline 1×)
- `matchedqa_byt5` (ByT5 QA-Upsampled 20×)
- `multitask_byt5` (ByT5 Multitask)
- `multitask_byt5_lora` (ByT5 LoRA)

These checkpoints are available in Google Drive from the original Colab training runs. Evaluation scripts are in `scripts/04_eval_predictions.py` with the `--config configs/matchedqa_byt5.yaml` flag; modify `eval.generation_max_new_tokens` to test each limit.

Upon completion, Tables 1–3 should be updated with verified values, and the "†" annotations removed from the ByT5 column.

---

## 9. Threats to Validity

**Construct validity.** Multitask NER supervision is operationalized as a specific tagged-output format. The observed effects (positive for mT5, negative for ByT5) may reflect the utility of this particular NER formulation, not of entity-aware supervision in general.

**Internal validity.** The translation baseline uses FLAN-T5-base without access to the passage (closed-book inference). This makes the fine-tuned models structurally advantaged, not merely architecturally superior. Comparison results should not be generalized to passage-reading translation pipelines.

**External validity.** Swahili is zero-shot for all fine-tuned models. Hausa and Yoruba provide the only trained evidence. Three languages from two language families (Afro-Asiatic and Niger-Congo) are insufficient for broad claims about African NLP.

**Statistical validity.** All results are based on a single random seed (42). No confidence intervals are available.

**Measurement validity.** All ByT5 QA metrics are lower bounds due to the 16-byte generation cap. The true ByT5 metrics are unknown until re-evaluation at corrected limits.

---

## 10. Limitations

1. **ByT5 generation cap.** The 16-byte cap constrains approximately 24% of ByT5 predictions. Corrected evaluation at 128 bytes is required before submission.
2. **Single random seed.** All results are based on seed 42. Variance across seeds is unknown.
3. **Swahili is zero-shot.** No Swahili QA examples appear in training. Swahili results represent zero-shot transfer, not low-resource fine-tuning.
4. **Translation baseline is closed-book.** FLAN-T5-base does not receive the passage; the comparison is not between equivalent QA systems.
5. **QA exposure not step-matched.** Multitask training takes 3.04× more optimizer steps per epoch than QA-upsampled training. Δ_task reflects equal QA exposure per epoch, not equal total compute.
6. **Language scope.** Three languages; generalization to other African languages is unverified.
7. **ByT5 checkpoints not committed.** ByT5 checkpoint files are not in the repository and must be accessed from Colab Drive for re-evaluation.

---

## 11. Future Work

1. **Re-evaluate ByT5 at corrected generation limits** (priority 1 before submission).
2. **Multi-seed validation** (≥3 seeds for top 3 configurations).
3. **Swahili training data.** Acquire or annotate Swahili QA training examples to replace zero-shot results with low-resource fine-tuning.
4. **Passage-reading translation baseline.** Replace FLAN-T5-base closed-book QA with an extractive reader on the translated passage (e.g., mBERT extractive QA on translated text) for a fair comparison.
5. **Epoch-volume ablation.** Train on 1× data for 200 epochs to separate epoch count from upsampling effects.
6. **ByT5 adapter placement study.** Vary LoRA rank and placement to identify why ByT5 adapters underperform.

---

## 12. Conclusion

We conducted a controlled ablation study of direct multilingual QA for low-resource African languages. The primary empirical finding — confirmed cleanly for mT5 and directionally for ByT5 (subject to generation-cap correction) — is that 20× QA upsampling moves direct models from substantially below a translation-based parametric baseline (mT5 EM 0.032 vs 0.116) to substantially above it (mT5 EM 0.197 vs 0.116). Gradient exposure volume is the dominant driver.

The secondary finding — that NER supervision helps mT5 (+0.013 EM) while harming ByT5 overall (−0.027† EM) — is theoretically interesting but limited by a single seed, a zero-shot Swahili condition, and suppressed ByT5 metrics. The finding is consistent across seven of nine language-metric comparisons, providing some directional robustness, but cannot be treated as statistically reliable without multi-seed validation and corrected ByT5 evaluation.

We release all code, predictions, training logs, and evaluation artifacts. ByT5 checkpoint re-evaluation at corrected generation limits is a required step before submission. See §8 for the re-evaluation protocol.

---

## References

Adelani, D. I., et al. (2022). MasakhaNER 2.0: Africa-centric Transfer Learning for Named Entity Recognition. In *Proceedings of EMNLP 2022*, pp. 4488–4508.

Artetxe, M., Ruder, S., & Yogatama, D. (2020). On the Cross-lingual Transferability of Monolingual Representations (XQuAD). In *Proceedings of ACL 2020*, pp. 4623–4637.

Caruana, R. (1997). Multitask Learning. *Machine Learning*, 28(1), 41–75.

Clark, J. H., et al. (2020). TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages. *TACL*, 8, 454–470.

Collobert, R., et al. (2011). Natural Language Processing (Almost) from Scratch. *JMLR*, 12, 2493–2537.

Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT Sentence Embedding. In *Proceedings of ACL 2022*, pp. 878–891.

Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. In *Proceedings of ICLR 2022*.

Longpre, S., Lu, Y., & Daiber, J. (2021). MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain QA. *TACL*, 9, 1389–1406.

Nakkiran, P., et al. (2020). Deep Double Descent: Where Bigger Models and More Data Hurt. In *Proceedings of ICLR 2020*.

NLLB Team. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. *arXiv:2207.04672*.

Ogundepo, O., et al. (2023). AfriQA: Cross-lingual Open-Retrieval QA for African Languages. In *Findings of EMNLP 2023*, pp. 14957–14972.

Sutton, R. (2019). The Bitter Lesson. *Blog post, incompleteideas.net*.

Xue, L., et al. (2021). mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer. In *Proceedings of NAACL 2021*, pp. 483–498.

Xue, L., et al. (2022). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. *TACL*, 10, 291–306.

*[CITATION NEEDED: Chung et al. 2022 — Scaling Instruction-Finetuned Language Models (FLAN-T5)]*

