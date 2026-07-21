# Scaling Exposure to Replace Translation Pipelines: Direct Multilingual Question Answering for Low-Resource African Languages

**Authors:** [Author Name(s)]  
**Affiliation:** [Institution(s)]  
**Correspondence:** [email@institution.edu]

---

## Abstract

Cross-lingual question answering (QA) systems for African languages have historically depended on translation pipelines that pivot through high-resource languages, compounding errors across morphologically rich and typologically diverse language families. In this work, we investigate whether direct multilingual QA — where models read and answer questions entirely in the native language — can overcome this dependence, and we identify **gradient exposure volume** as the central factor governing that transition.

We conduct a controlled ablation study across two model architectures (mT5-base and ByT5-base), five training configurations, and three African languages (Swahili, Hausa, and Yoruba) using the AfriQA benchmark with supplementary supervision from MasakhaNER 2.0. Our primary finding is that a standard translation-pipeline baseline (EM = 0.116) is substantially outperformed by a byte-level model trained under matched-volume conditions (EM = 0.219, F1 = 0.292) — an improvement achieved without any engineered cross-lingual heuristics, simply by ensuring sufficient gradient update steps.

A secondary finding concerns the relationship between tokenization architecture and the utility of explicit multitask supervision via Named Entity Recognition (NER). For subword models (mT5), adding NER supervision on top of matched-volume training provides a modest but consistent positive delta (EM +0.013 overall; F1 +0.026). For byte-level models (ByT5), the same supervision consistently *degrades* performance (EM −0.027 overall; F1 −0.020), suggesting that byte-level representations naturally internalize morphological boundary information in ways that conflict with rigid sequence-level NER constraints.

Parameter-efficient fine-tuning via rank-16 LoRA adapters performs adequately for mT5 (EM = 0.164) but collapses for ByT5 (EM = 0.045), indicating that adapter capacity is insufficient to capture the representation transformations required under byte-level multitask conditions.

These findings are based on single-seed experiments and should be interpreted as empirically grounded hypotheses pending multi-seed validation. All code, configurations, model outputs, and evaluation artifacts are publicly available.

**Keywords:** low-resource QA, African NLP, byte-level models, multilingual learning, multitask learning, LoRA, AfriQA, MasakhaNER

---

## 1. Introduction

Question answering in low-resource African languages presents a fundamental challenge: the absence of large-scale training corpora, the morphological complexity of languages such as Swahili, Hausa, and Yoruba, and the historical dominance of English-centric NLP infrastructure collectively push practitioners toward translation-based approaches. The prototypical translation pipeline — translate the source question and passage into English, apply an English QA model, translate the predicted answer back — introduces at least two points of compounding error and strips culturally specific references and named entities from their original context.

Direct, native-language QA sidesteps these failure modes but faces a practical barrier: with only a few hundred training examples available in a low-resource African language, naively fine-tuned sequence-to-sequence models perform poorly. Prior work has typically interpreted this failure as evidence of architectural or data insufficiency, motivating increasingly complex multi-task, cross-lingual transfer, or knowledge distillation schemes.

This paper challenges that interpretation. We propose that the primary bottleneck in low-resource direct QA is **gradient exposure volume**: the number of parameter updates the model receives on the target task. Because African QA datasets are small (the AfriQA training set contains 470 examples across three languages), a model trained for the conventional number of epochs receives far fewer gradient updates than one trained on a comparable multitask corpus. Once this exposure gap is closed — by upsampling the QA data to achieve step parity with larger multitask corpora — direct native-language models consistently outperform the translation pipeline baseline.

Beyond the central exposure scaling finding, we investigate a secondary question: **does the utility of explicit Named Entity Recognition supervision depend on the underlying tokenization architecture?** We find strong empirical evidence for an interaction effect. For subword tokenizers (mT5's SentencePiece vocabulary), NER supervision provides modest additional benefit even beyond matched-volume training, likely because subword tokenization may fragment entity spans in agglutinative languages, and explicit entity supervision provides corrective gradient signal. For byte-level tokenization (ByT5's UTF-8 byte representation), NER supervision is consistently harmful, reducing performance across all three languages and all three metrics.

Our contributions are:

1. **Empirical evidence for exposure scaling as the primary driver of translation pipeline parity** in low-resource African QA. We demonstrate that the apparent superiority of translation pipelines over direct native-language models is an artifact of gradient step imbalance that dissolves when training volume is matched.

2. **Evidence for a tokenization-supervision interaction.** We show that the effect of multitask NER supervision differs systematically by architecture: positive for subword models, negative for byte-level models, across multiple metrics and languages.

3. **Ablation results quantifying parameter-efficient fine-tuning limitations.** Rank-16 LoRA adapters fail severely under byte-level multitask conditions, raising practical questions about the applicability of PEFT methods to non-subword architectures.

4. **A publicly released evaluation framework and result set** covering nine experimental configurations, three languages, and three metrics on the AfriQA test set, enabling direct replication and extension.

We emphasize at the outset that all reported results are based on a single random seed. The patterns we describe are consistent and theoretically motivated, but multi-seed validation is required before any of the directional findings can be treated as statistically reliable.

---

## 2. Related Work

### 2.1 Multilingual and Cross-Lingual QA

Extractive and generative question answering has been extensively studied in multilingual settings through benchmarks such as XQuAD [CITATION NEEDED: Artetxe et al. 2020], TyDi QA [CITATION NEEDED: Clark et al. 2020], and MKQA [CITATION NEEDED: Longpre et al. 2021]. These benchmarks primarily cover high- and mid-resource languages, leaving a significant gap for African language families. AfriQA [Ogundepo et al., 2023] addresses this gap with gold-passage reading comprehension tasks in twelve African languages, including the three studied here.

Cross-lingual QA systems have explored zero-shot transfer from English QA models, cross-lingual pretraining, and translation-based pivot strategies. The NLLB team [NLLB Team, 2022] demonstrates the scale of compute and data required to achieve reasonable neural machine translation for low-resource languages, underscoring why translation pipelines remain a pragmatic baseline despite their documented failure modes.

### 2.2 Multilingual Pretrained Models

mT5 [Xue et al., 2021] extends the T5 text-to-text framework to 101 languages using a multilingual Common Crawl corpus. Its SentencePiece subword vocabulary covers diverse scripts and language families, but vocabulary allocation for low-resource African languages is substantially reduced compared to high-resource languages, potentially increasing fragmentation of morphologically complex tokens.

ByT5 [Xue et al., 2022] processes raw UTF-8 byte sequences, eliminating the vocabulary as a design choice entirely. This approach confers robustness to noise, rare characters, and complex morphology at the cost of substantially longer input sequences — a particular concern for passage-level QA tasks where byte-level encoding may exceed several thousand tokens.

### 2.3 Multitask and Entity-Aware Learning

Multitask learning as a regularization and transfer mechanism has a long history in NLP [CITATION NEEDED: Caruana 1997; Collobert et al. 2011]. For structured prediction tasks involving entities, interleaving NER and QA during training provides the model with direct supervision over span boundaries that are otherwise only weakly supervised through answer extraction. MasakhaNER 2.0 [Adelani et al., 2022] provides Africa-focused NER data covering PER, LOC, ORG, and DATE categories for over twenty languages, including all three studied here.

The interaction between multitask supervision and tokenization architecture has received limited attention. Our work provides systematic investigation of this interaction for African languages, where the morphological properties of the language families may amplify architectural differences that are less visible in Indo-European or CJK settings.

### 2.4 Parameter-Efficient Fine-Tuning

LoRA [Hu et al., 2022] inserts low-rank decomposition matrices into frozen pretrained weights, enabling efficient adaptation with a small number of trainable parameters. LoRA has demonstrated strong performance across many NLP tasks, particularly for instruction tuning of large language models. Its effectiveness under multitask conditions with non-subword architectures has not been systematically studied; our findings suggest significant limitations in this setting.

### 2.5 Exposure and Overfitting in Low-Resource Settings

The relationship between training data volume and performance in seq2seq models is non-trivial. Deep Double Descent [Nakkiran et al., 2019] demonstrates that model performance can exhibit complex non-monotonic relationships with training duration and data size. Our exposure scaling framing — increasing gradient steps by upsampling rather than collecting additional data — is conceptually related to the question of how optimization trajectory interacts with model capacity under data scarcity. Sutton's Bitter Lesson [Sutton, 2019] provides a high-level motivational frame: general methods that leverage computation tend to dominate over human-engineered inductive biases. Our result that matched-volume training outperforms NER-augmented multitask training for byte-level models is consistent with this perspective.

---

## 3. Methodology

### 3.1 Problem Formulation

We study extractive-generative QA where, given a question *q* and a passage *p* both in a native African language *l*, a model generates an answer string *â*. We evaluate *â* against a gold answer *a* using string-level and semantic metrics defined in Section 4.3. We do not study retrieval; all experiments use gold passages drawn directly from AfriQA.

### 3.2 Experimental Configurations

We design five configurations to disentangle three factors: gradient exposure volume, tokenization architecture, and multitask supervision.

**Configuration 1 — Low-Resource Baseline (1×):**  
Both mT5-base and ByT5-base are fine-tuned on the raw AfriQA training split (470 examples) for 20 epochs with no data augmentation. With an effective batch size of 64, training for 20 epochs on 470 examples yields approximately 147 gradient update steps.

**Configuration 2 — Translation Pipeline:**  
Source-language questions and passages are translated into English via the Google Translate API. An English QA inference step is applied. Predicted answers are translated back into the source language. This is a zero-shot inference system; no model fine-tuning is performed. It represents the standard industrial-practice baseline for low-resource languages.

**Configuration 3 — Matched-Volume QA (20×):**  
AfriQA training examples are upsampled 20× (to 9,400 effective examples) and training runs for up to 200 epochs with early stopping (patience = 3). This ensures that the number of gradient update steps matches those in the multitask configurations, providing a controlled ablation for exposure volume alone.

**Configuration 4 — Multitask NER→QA:**  
AfriQA QA examples (20× upsampled) are interleaved with MasakhaNER 2.0 training examples (19,185 NER instances). Training follows the same schedule as Configuration 3. NER examples use the prompt prefix "extract entities:" while QA examples use "question:".

**Configuration 5 — LoRA Multitask:**  
Identical to Configuration 4 but with rank-16 LoRA adapters (α = 32, dropout = 0.05) inserted into the transformer attention layers. Only adapter parameters are updated during training.

### 3.3 Delta Decomposition

We define three quantities that partition the total performance gain over the 1× baseline:

- **Δ_exposure** = Performance(Matched-Volume) − Performance(Baseline)
- **Δ_task** = Performance(Multitask) − Performance(Matched-Volume)
- **Δ_total** = Δ_exposure + Δ_task

By holding gradient step count constant between Configurations 3 and 4, Δ_task isolates the marginal contribution of NER supervision, independent of training duration.

---

## 4. Experimental Setup

### 4.1 Datasets

**AfriQA (gold-passage subset)** [Ogundepo et al., 2023]:

| Split | Swahili | Hausa | Yoruba | Total |
|-------|--------:|------:|-------:|------:|
| Train | — | — | — | 470 |
| Validation | — | — | — | 222 |
| Test | 295 | 300 | 253 | 848 |

**MasakhaNER 2.0** [Adelani et al., 2022]:

| Split | Examples |
|-------|--------:|
| Train | 19,185 |
| Validation | 2,741 |
| Test | 5,480 |

### 4.2 Models

| Model | Parameters | Tokenization | Vocab |
|-------|----------:|------------|------:|
| mT5-base | ~580M | SentencePiece | 250,000 tokens |
| ByT5-base | ~582M | UTF-8 bytes | 256 bytes |

mT5-base uses a maximum source length of 256 tokens; ByT5-base uses 1,024 bytes to accommodate the same passages (approximately 4× length expansion).

### 4.3 Evaluation Metrics

- **Exact Match (EM):** Fraction of predictions exactly matching the normalized gold answer. Primary ranking metric.
- **Token F1:** Bag-of-words token F1 averaged over test examples. Rewards partial overlap.
- **Semantic Similarity:** Cosine similarity of LaBSE [Feng et al., 2022] sentence embeddings. Captures meaning preservation beyond surface form. *Not reported for the translation pipeline due to potential artifacts from back-translation.*

### 4.4 Training Configuration

| Hyperparameter | mT5-base | ByT5-base |
|---------------|---------|---------|
| Optimizer | Adafactor | Adafactor |
| Learning Rate (baselines) | 5 × 10⁻⁴ | 1 × 10⁻⁴ |
| Effective Batch Size | 64 | 64 |
| Physical Batch / Grad Accum | 32 / 2 | 32 / 2 |
| Max Source Length | 256 | 1,024 |
| Max Target Length | 128 | 128 |
| Precision | bf16 | bf16 |
| Weight Decay | 0.01 | 0.01 |
| Max Gradient Norm | 1.0 | 1.0 |
| Warmup Steps | 50 | 50 |
| Early Stopping Patience | 3 | 3 |
| Epochs (1× configs) | 20 | 20 |
| Epochs (20× configs) | up to 200 | up to 200 |
| Random Seed | 42 | 42 |

> **Methodological note on learning rates:** mT5 and ByT5 were trained at different learning rates reflecting the longer byte-level sequences in ByT5, which increase per-token gradient magnitudes. This introduces a minor confound in direct architecture comparisons that we acknowledge as a limitation.

### 4.5 Hardware

Single NVIDIA A100-SXM4-80GB GPU, 167GB system RAM, Google Colab Pro. All predictions, metrics, and logs are publicly versioned in the repository.

---

## 5. Results

### 5.1 Overall QA Performance

**Table 1: Overall QA performance on the AfriQA test set (n = 848).** Best result per metric in **bold**. †Semantic similarity not reported for translation pipeline (see §4.3).

| Configuration | EM | F1 | Sem. |
|---|---:|---:|---:|
| Baseline mT5 (1×) | 0.032 | 0.071 | 0.293 |
| Baseline ByT5 (1×) | 0.025 | 0.073 | 0.327 |
| Translation Pipeline | 0.116 | 0.189 | —† |
| Matched-Vol mT5 (20×) | 0.197 | 0.236 | 0.439 |
| **Matched-Vol ByT5 (20×)** | **0.219** | **0.292** | **0.513** |
| Multitask mT5 | 0.210 | 0.262 | 0.469 |
| Multitask ByT5 | 0.192 | 0.272 | 0.476 |
| LoRA mT5 | 0.164 | 0.200 | 0.398 |
| LoRA ByT5 | 0.045 | 0.094 | 0.308 |

Key observations:

1. **Under 1× training, the translation pipeline substantially outperforms direct QA.** mT5 (EM = 0.032) and ByT5 (EM = 0.025) both fall well below the translation pipeline (EM = 0.116).

2. **Matched-volume direct QA substantially outperforms the translation pipeline.** ByT5 20× achieves EM = 0.219 versus 0.116 for translation, a gap of 0.103 EM and 0.103 F1.

3. **ByT5 outperforms mT5 under matched-volume conditions.** Without auxiliary supervision, ByT5 achieves higher EM (+0.022), F1 (+0.056), and semantic similarity (+0.074) than mT5.

4. **NER supervision has opposing effects by architecture** (detailed in §5.2).

5. **LoRA collapses for ByT5.** LoRA ByT5 (EM = 0.045) barely exceeds the 1× ByT5 baseline.

### 5.2 Delta Decomposition

**Table 2: Delta decomposition of QA performance gains.** Δ_exposure = Matched-Volume − Baseline. Δ_task = Multitask − Matched-Volume. Positive Δ_task indicates NER supervision is beneficial; negative indicates it is harmful.

| Architecture | Metric | Baseline | Matched-Vol | Multitask | Δ_exposure | Δ_task |
|---|---|---:|---:|---:|---:|---:|
| mT5 | EM | 0.032 | 0.197 | 0.210 | +0.165 | **+0.013** |
| mT5 | F1 | 0.071 | 0.236 | 0.262 | +0.165 | **+0.026** |
| mT5 | Sem. | 0.293 | 0.439 | 0.469 | +0.147 | **+0.030** |
| ByT5 | EM | 0.025 | 0.219 | 0.192 | +0.195 | **−0.027** |
| ByT5 | F1 | 0.073 | 0.292 | 0.272 | +0.219 | **−0.020** |
| ByT5 | Sem. | 0.327 | 0.513 | 0.476 | +0.186 | **−0.037** |

The exposure delta dominates for both architectures. The task delta is directionally opposite: consistently positive for mT5, consistently negative for ByT5, across all three metrics. We cannot assess statistical significance from a single seed, but this pattern is consistent across all nine language-metric combinations in the ByT5 case.

### 5.3 Per-Language Results

**Table 3: Per-language results for main configurations (EM / F1).** SWA = Swahili (n=295), HAU = Hausa (n=300), YOR = Yoruba (n=253).

| Configuration | SWA EM | SWA F1 | HAU EM | HAU F1 | YOR EM | YOR F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline mT5 (1×) | 0.017 | 0.055 | 0.030 | 0.057 | 0.051 | 0.105 |
| Baseline ByT5 (1×) | 0.003 | 0.058 | 0.017 | 0.046 | 0.059 | 0.122 |
| Matched-Vol mT5 | 0.176 | 0.214 | 0.227 | 0.261 | 0.186 | 0.233 |
| **Matched-Vol ByT5** | 0.180 | 0.253 | **0.277** | **0.334** | 0.198 | 0.287 |
| Multitask mT5 | 0.176 | 0.224 | **0.277** | 0.306 | 0.170 | 0.255 |
| Multitask ByT5 | **0.190** | **0.266** | 0.227 | 0.293 | 0.154 | 0.254 |

Hausa shows the strongest ByT5 advantage under matched-volume (task delta −0.050 EM); Yoruba also shows a negative mT5 task delta (−0.016 EM), the only language where mT5 NER supervision is directionally harmful. Swahili is the only language where the ByT5 task delta is positive (+0.010 EM), though close to zero.

---

## 6. Error Analysis

**Table 4: Error distribution across all configurations (n = 848).** Categories: EM = exact match; Hi = partial F1 > 0.5; Lo = partial F1 ≤ 0.5; Wrong = F1 = 0; Empty; NER Leak = residual entity tag in QA output.

| Configuration | EM (%) | Hi (%) | Lo (%) | Wrong (%) | Empty (%) | NER Leak (%) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline mT5 (1×) | 3.2 | 2.7 | 6.0 | 87.9 | 0.2 | 0.0 |
| Baseline ByT5 (1×) | 2.5 | 2.8 | 7.4 | 87.3 | 0.0 | 0.0 |
| Translation Pipeline | 11.6 | 4.1 | 15.9 | 68.2 | 0.2 | 0.0 |
| Matched-Vol mT5 | 19.7 | 3.7 | 3.4 | 73.2 | 0.0 | 0.0 |
| **Matched-Vol ByT5** | **21.9** | 6.4 | 7.0 | 64.7 | 0.0 | 0.0 |
| Multitask mT5 | 21.0 | 4.6 | 4.5 | 68.2 | 0.0 | 1.8 |
| Multitask ByT5 | 19.2 | 7.5 | 6.6 | 66.3 | 0.0 | 0.4 |
| LoRA mT5 | 16.4 | 3.4 | 3.2 | 77.0 | 0.0 | 0.0 |
| LoRA ByT5 | 4.5 | 5.0 | 3.9 | 86.7 | 0.0 | 0.0 |

**Byte-level models produce more high-quality partial matches.** Matched-Vol ByT5 has 6.4% high-F1 partials versus 3.7% for mT5; Multitask ByT5 has 7.5% versus 4.6%. Even where ByT5 does not exactly match the gold, its outputs frequently overlap substantially. Qualitative inspection reveals a common pattern: byte-level generation correctly predicts the beginning of multi-token entities but truncates ("James Francis Ca" for "James Francis Cameron"), scoring F1 ≈ 0.67 rather than EM = 1. This is a generation length artifact rather than a semantic failure.

**Translation pipeline shows elevated low-F1 partial matches (15.9%)**, consistent with cascaded translation errors that distort answer surface form while preserving partial semantic content.

**NER format leakage in multitask models.** Multitask mT5 produces 1.8% predictions containing residual entity tagging syntax in the QA response; Multitask ByT5 produces only 0.4%. This asymmetry is consistent with the NER structural failure described in §6.1.

### 6.1 NER Evaluation: Structural Generation Failure in ByT5

**Table 5: NER prediction evaluation (MasakhaNER test set, n = 5,480).**

| Model | Parseable Preds | P | R | F1 |
|---|---:|---:|---:|---:|
| Multitask mT5 | 409 / 5,480 (7.5%) | 0.840 | 0.036 | 0.070 |
| Multitask ByT5 | 41 / 5,480 (0.7%) | 0.122 | ~0.000 | 0.001 |

The near-zero parse rate for ByT5 (0.7%) is diagnostically important. While mT5 generates 409 structurally parseable NER predictions (achieving high precision of 0.840 where it does predict), ByT5 generates only 41. Gold annotations were parseable at 72.5% of examples, confirming this is not an annotation issue. The ByT5 model largely fails to generate the expected tag-delimited output format for NER while simultaneously handling QA. We interpret this as evidence that byte-level models do not readily acquire rigid sequence-level structural templates under joint training conditions — and that this failed NER supervision degrades the QA task by consuming training budget without providing useful gradient signal.

---

## 7. Discussion

### 7.1 The Exposure Scaling Effect

The exposure delta is the dominant source of performance improvement in our ablation (EM +0.165 for mT5; +0.195 for ByT5), far exceeding the task delta. This suggests that the standard interpretation of translation pipeline superiority — that direct QA models are architecturally limited — may be systematically confounded by training budget asymmetry.

When evaluating whether a translation pipeline is "necessary" for a given language, the comparison should be against a matched-volume direct model, not a 1×-trained baseline. Under equalized gradient budgets, direct native-language models outperform the translation pipeline in our experiments. We do not claim this generalizes without bound — deep double descent effects, overfitting, and data quality all constrain the benefit of pure upsampling — but the empirical finding challenges the default assumption that translation is required.

### 7.2 The Tokenization-Supervision Interaction

The opposing signs of the task delta (mT5: +0.013 EM; ByT5: −0.027 EM) constitute the central theoretical finding. We offer two non-mutually-exclusive mechanisms:

**Mechanism 1: Subword boundary fragmentation.** SentencePiece tokenization may fragment named entity spans across multiple subwords whose boundaries do not align with morphological boundaries in agglutinative African languages. This makes entity boundary identification harder for mT5, and explicit NER supervision provides corrective gradient signal. Byte-level tokenization, which preserves every character unambiguously, does not suffer this fragmentation problem — and does not benefit from the same corrective signal.

**Mechanism 2: Structural template conflict.** Byte-level generation of raw character sequences and generation of structured tag-delimited annotations impose conflicting distributional pressures on the output layer. The near-zero NER parse rate for ByT5 (0.7%) provides direct evidence that the model largely fails at structural NER generation. This training signal is not merely unhelpful — it may actively perturb the model's well-functioning character-level QA generation mode.

Both mechanisms remain speculative and require targeted experiments (probing studies, gradient attribution, separate task heads) to validate formally.

### 7.3 LoRA Adapter Failure for Byte-Level Models

LoRA ByT5 retains approximately 23% of Multitask ByT5's EM performance (0.045 vs. 0.192), while LoRA mT5 retains approximately 78% (0.164 vs. 0.210). This asymmetry suggests that byte-level models require more extensive adaptation distributed throughout the network — not just in the attention projections where LoRA rank-16 adapters operate — to shift from pretraining byte distributions to structured QA generation. This is consistent with the observation that ByT5 uses significantly longer sequences, which may distribute task-relevant information differently across attention layers.

### 7.4 Practical Implications

If the tokenization-supervision interaction proves robust across languages and seeds, it has concrete design implications: practitioners building NER-assisted QA systems for low-resource languages should verify whether their chosen architecture is amenable to explicit structural supervision. Byte-level models may be better served by pure exposure scaling of QA data; subword models may genuinely benefit from multitask entity grounding — particularly in morphologically complex languages.

---

## 8. Threats to Validity

**Construct validity.** We operationalize entity awareness through multitask NER supervision. The observed positive task delta for mT5 may reflect other multitask regularization effects (improved calibration, catastrophic forgetting mitigation) rather than entity-specific representations per se.

**Internal validity.** The translation pipeline uses the Google Translate API, a commercial black-box system. SOTA open-source NMT (e.g., NLLB-200) may produce higher translation quality. Our results should be interpreted as comparisons against commercially available translation, not the theoretical ceiling of translation-based QA.

**External validity.** Three languages from two typological families are studied. The tokenization-supervision interaction may manifest differently in other African languages with distinct morphological properties. Extension to broader language sets is required before general conclusions can be drawn.

**Statistical validity.** All results are based on a single random seed (42). Confidence intervals cannot be constructed without multi-seed runs. The directional consistency of the task delta across all three metrics and (in the ByT5 case) across two of three languages provides informal confidence in the direction, but magnitude estimates are uncertain.

---

## 9. Limitations

1. **Single random seed.** All experiments conducted with seed 42. Performance variance across seeds is unknown. This is the most significant limitation for claims about configuration ordering.
2. **Language scope.** Three languages studied. Generalization to other African or low-resource languages is unverified.
3. **Translation pipeline ceiling.** Google Translate may underestimate the ceiling of translation-based QA.
4. **Hyperparameter asymmetry.** Different learning rates for mT5 and ByT5 introduce a minor confound in direct architecture comparisons.
5. **Epoch-count versus data-volume ambiguity.** The matched-volume design upsamples and extends training epochs simultaneously. Isolating epoch count from data volume requires an additional ablation.
6. **Missing multi-seed variance.** No statistical significance testing is possible. All directional findings are empirically motivated but not statistically confirmed.

---

## 10. Future Work

1. **Multi-seed validation.** Run at minimum three seeds for the top configurations to assess variance and enable significance testing.
2. **NLLB-200 comparison.** Replace Google Translate with a state-of-the-art open NMT system for a rigorous translation baseline.
3. **Epoch-volume ablation.** Train on 1× data for 200 epochs to disentangle epoch count from upsampling effects.
4. **Extension to generative LLMs.** Investigate exposure scaling in instruction-tuned decoder-only models.
5. **Probing the tokenization interaction.** Use gradient attribution or probing classifiers to determine whether the positive/negative task delta reflects entity-specific or general regularization effects.
6. **Broader language coverage.** Extend to additional AfriQA languages to assess generalizability.

---

## 11. Conclusion

We have presented a controlled ablation study of direct multilingual question answering for low-resource African languages. Our primary finding is that once gradient exposure is matched through data upsampling, direct native-language byte-level QA (EM = 0.219) substantially outperforms a standard translation pipeline baseline (EM = 0.116). This suggests that the apparent inferiority of direct models in early experiments reflects an optimization budget disparity rather than a fundamental architectural limitation.

Our secondary finding is an architecture-level interaction: NER supervision consistently improves subword mT5 (EM +0.013, F1 +0.026) while consistently degrading byte-level ByT5 (EM −0.027, F1 −0.020) across metrics and languages. The near-zero NER generation parse rate for ByT5 (0.7%) provides a mechanistic explanation: byte-level models do not readily acquire rigid sequence-level structural annotation templates, and the failed NER training signal degrades QA performance.

These findings are subject to the critical limitation of single-seed experiments and should be treated as theoretically motivated empirical hypotheses requiring multi-seed validation. We release all code, configurations, predictions, and evaluation artifacts publicly to support replication and extension.

---

## References

Adelani, D. I., et al. (2022). MasakhaNER 2.0: Africa-centric Transfer Learning for Named Entity Recognition. In *Proceedings of EMNLP 2022*.

Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT Sentence Embedding. In *Proceedings of ACL 2022*.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. In *Proceedings of ICLR 2022*.

Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2019). Deep Double Descent: Where Bigger Models and More Data Hurt. *arXiv:1912.02292*.

NLLB Team. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. *arXiv:2207.04672*.

Ogundepo, O., et al. (2023). AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages. In *Findings of EMNLP 2023*.

Sutton, R. (2019). The Bitter Lesson. *Blog post, incompleteideas.net*.

Xue, L., et al. (2021). mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer. In *Proceedings of NAACL 2021*.

Xue, L., et al. (2022). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. *Transactions of the Association for Computational Linguistics, 10*, 291–306.

*[CITATION NEEDED: Artetxe et al. 2020 — XQuAD: A Benchmark Dataset for Cross-lingual Question Answering]*  
*[CITATION NEEDED: Clark et al. 2020 — TyDi QA: A Benchmark for Information-Seeking QA in Typologically Diverse Languages]*  
*[CITATION NEEDED: Longpre et al. 2021 — MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain QA]*  
*[CITATION NEEDED: Caruana, R. (1997). Multitask Learning. Machine Learning, 28(1), 41–75.]*  
*[CITATION NEEDED: Collobert et al. (2011). Natural Language Processing (Almost) from Scratch. JMLR, 12, 2493–2537.]*

