# Eliminating Translation Pipelines in Multilingual Question Answering for African Languages

## 1. Introduction
Cross-lingual question answering (QA) systems often rely on translation-heavy pipelines that can amplify errors for low-resource African languages. This project investigates direct multilingual QA over Swahili, Hausa, and Yoruba passages, with the goal of improving answer extraction quality without translation intermediaries.

Our central idea is entity-aware transfer: first train on named entity extraction, then adapt to QA. We test whether sequential NER -> QA multitask training produces better answer quality than a QA-only baseline, and whether ByT5 offers additional benefits compared with mT5.

---

## 2. Methodology

### 2.1 Datasets and Languages
- **AfriQA** subset (`swa`, `hau`, `yor`) for QA.
- **MasakhaNER 2.0** subset (`swa`, `hau`, `yor`) for entity supervision.
- From run logs: AfriQA splits are train=470, validation=222, test=848.

### 2.2 Baseline: QA-only mT5
Baseline training uses `google/mt5-base` on QA examples only:
- **Input:** `question: {question} context: {passage}`
- **Target:** `{gold_answer}`

### 2.3 Entity-aware sequential multitask training
Multitask data mixes QA and NER formats:
- QA task: `question: ... context: ...`
- NER task: `extract entities: ...`

The actual training procedure in logs is **sequential**, not simultaneous:
1. Phase 1: train on NER-oriented multitask corpus
2. Phase 2: continue training for QA

Two sequential multitask models were evaluated:
- `google/mt5-base`
- `google/byt5-base`

### 2.4 Evaluation protocol
Evaluation files indicate three metrics:
1. **Exact Match (EM)**
2. **Token F1**
3. **Semantic similarity** (LaBSE cosine similarity)

For multitask prediction files, evaluation applies `--qa_only`, filtering out `lang=unknown` NER rows before QA scoring (848 QA rows retained).

---

## 3. Results and Evaluation (Repository-Grounded)

All values below are taken from:
- `outputs_colab/metrics/baseline_mt5_test_metrics.json`
- `outputs_colab/metrics/multitask_mt5_test_qa_only_metrics.json`
- `outputs_colab/metrics/multitask_byt5_test_qa_only_metrics.json`

### 3.1 Overall QA results (n=848)

| Model | EM | Token F1 | Semantic |
|---|---:|---:|---:|
| Baseline mT5 (QA-only) | 0.0318 | 0.0708 | 0.2925 |
| Sequential Multitask mT5 | 0.2099 | 0.2623 | 0.4691 |
| Sequential Multitask ByT5 | 0.1922 | 0.2717 | 0.4762 |

### 3.2 Per-language QA results

| Language | Baseline EM/F1/Sem | Multitask mT5 EM/F1/Sem | Multitask ByT5 EM/F1/Sem |
|---|---|---|---|
| **Swahili (swa)** | 0.0169 / 0.0552 / 0.2721 | 0.1763 / 0.2243 / 0.4258 | **0.1898 / 0.2656 / 0.4547** |
| **Hausa (hau)** | 0.0300 / 0.0573 / 0.2879 | **0.2767 / 0.3057 / 0.5131** | 0.2267 / 0.2929 / **0.5150** |
| **Yoruba (yor)** | 0.0514 / 0.1050 / 0.3218 | **0.1700 / 0.2550 / 0.4674** | 0.1542 / 0.2537 / 0.4552 |

### 3.3 Delta analysis

**Sequential multitask mT5 vs baseline**
- EM: +0.1781
- F1: +0.1915
- Semantic: +0.1766

**Sequential multitask ByT5 vs baseline**
- EM: +0.1604
- F1: +0.2010
- Semantic: +0.1837

**ByT5 vs multitask mT5**
- EM: -0.0177
- F1: +0.0095
- Semantic: +0.0071

Interpretation: both multitask systems strongly beat baseline; ByT5 trades slightly lower EM for slightly higher F1/semantic compared with multitask mT5.

---

## 4. Hypothesis Evaluation

### 4.1 Main hypothesis (sequential NER -> QA improves QA)
Supported by current outputs. Relative to baseline, both sequential multitask models produce large gains across all three metrics and all three languages.

### 4.2 Secondary hypothesis (ByT5 may further help)
Partially supported:
- ByT5 improves overall F1 and semantic similarity over multitask mT5.
- ByT5 is lower on overall EM and is not uniformly better per language (especially Hausa and Yoruba EM).

### 4.3 Does entity-aware training help semantic understanding more than exact matching?
Evidence is mixed but favorable:
- Compared with baseline, multitask models improve semantic similarity substantially.
- ByT5's improvements over multitask mT5 are concentrated in F1/semantic rather than EM.
- This pattern is compatible with better semantic matching despite imperfect exact-span reproduction.

---

## 5. LoRA Status (What is and is not supported)

A LoRA workflow script exists (`run_compare_lora.sh`), but there are **no LoRA metrics/prediction artifacts** in the refreshed evaluation outputs:
- No `multitask_mt5_lora_test.jsonl`
- No LoRA metric JSON/CSV in `outputs_colab/metrics`

Therefore, no empirical LoRA-vs-full conclusion can be made from the currently available evaluation files.

---

## 6. Caveats and Potential Confounds

1. **Training-data imbalance confound:** baseline uses QA-only train=470, while multitask uses a much larger combined corpus (train=28,585 after upsampling), so gains cannot be attributed solely to entity-awareness without additional controls.
2. **Model-comparison confound:** logs show different max source lengths for mT5 and ByT5 runs (256 vs 1024), which may influence outcomes.
3. **Single-run limitation:** current artifacts do not include multi-seed statistics or confidence intervals.
4. **Metric implementation note:** token F1 in code uses set-style token overlap; this can affect absolute F1 calibration, though within-repo comparisons remain consistent.

---

## 7. Strongest Defensible Conclusion

From the current repository outputs, the strongest defensible claim is:

**Sequential multitask training with NER -> QA is associated with substantial QA quality gains over the QA-only mT5 baseline on AfriQA (Swahili, Hausa, Yoruba), across EM, token F1, and semantic similarity.**

A stronger causal claim that entity-awareness alone drives all gains requires controlled ablations (matched data volume, steps, and context length) and multi-seed validation.

---

## 8. Recommended Next Experiments

1. **Controlled ablation:** QA-only vs multitask-sequential with matched total training budget.
2. **ByT5 fairness run:** equalize max source length and compute budget vs mT5.
3. **Multi-seed replication:** report mean/std and paired significance.
4. **LoRA evaluation completion:** generate and archive LoRA predictions/metrics to enable full vs LoRA comparison.
5. **Mechanism probe:** evaluate NER quality directly and correlate NER improvements with QA gains.

---

## 9. Updated Discussion Paragraph (for Results section)

On the AfriQA test split (n=848 QA examples across Swahili, Hausa, and Yoruba), sequential multitask training (NER -> QA) substantially outperformed the QA-only mT5 baseline. Baseline performance was low (EM/F1/Semantic = 0.0318/0.0708/0.2925), while sequential multitask mT5 reached 0.2099/0.2623/0.4691 and sequential multitask ByT5 reached 0.1922/0.2717/0.4762. Gains versus baseline were consistent across all three languages. Compared with multitask mT5, ByT5 improved F1 and semantic similarity but reduced EM, indicating a trade-off between strict exact matching and softer semantic fidelity. These results strongly support a directional benefit of sequential multitask adaptation for multilingual African-language QA, while controlled ablations and multi-seed evidence are still needed for a conference-grade causal claim about entity-awareness.
