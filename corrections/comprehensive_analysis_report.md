# Comprehensive Research Analysis Report
## AfriQA Entity-Aware Multilingual QA

**Generated**: 2026-07-08  
**Purpose**: Research review and gap analysis — no code modifications made.

---

## 1. Executive Summary

This repository implements a **direct, translation-free question answering system** for three low-resource African languages (Swahili, Hausa, Yoruba) using seq2seq models (mT5, ByT5). The central research question is whether **entity-aware multitask learning** (joint QA + NER training) can replace traditional translation-heavy XOR QA pipelines.

Through a well-designed ablation study, the project discovered that:
1. **Training exposure volume** (data upsampling) accounts for >90% of the performance gains originally attributed to NER supervision.
2. **Byte-level models (ByT5)** natively handle entity boundaries without NER assistance, while **subword models (mT5)** still benefit from explicit entity supervision.
3. **LoRA adapters** collapse under complex multitask regimes, especially for ByT5.

The project has **7 completed experiment configurations** with stored metrics, a well-structured codebase, and a paper draft. However, significant gaps remain before the work reaches the threshold for a strong venue publication.

---

## 2. Repository Architecture

### 2.1 Project Structure

```
afriqa-entity-aware-qa/
├── src/afriqa_ner_qa/          # Core library (6 modules)
│   ├── config.py               # YAML config loader
│   ├── paths.py                # ProjectPaths dataclass
│   ├── logging_utils.py        # Logger setup
│   ├── data.py                 # Data loading, NER/QA formatting, JSONL export
│   ├── train.py                # Tokenization, Seq2SeqTrainer builder
│   └── eval.py                 # EM, F1, LaBSE semantic similarity, prediction generation
├── scripts/                    # Executable pipeline scripts (8 files)
│   ├── 00_download_and_subset.py
│   ├── 01_prepare_qa_data.py
│   ├── 01b_prepare_multitask_data.py
│   ├── 02_train_baseline_qa.py
│   ├── 02b_eval_translation_pipeline.py
│   ├── 03_train_multitask_qa.py
│   ├── 04_eval_predictions.py
│   └── pull_drive_results.py
├── configs/                    # YAML experiment configs (3 files)
│   ├── default.yaml
│   ├── matchedqa_mt5.yaml
│   └── matchedqa_byt5.yaml
├── run_all_experiments.sh      # End-to-end runner
├── run_compare_lora.sh         # Full vs LoRA comparison
├── outputs_colab/              # Archived experiment results
├── data/processed/             # Prepared datasets
├── Research Papers/            # 3 reference papers
├── paper_draft.md              # Draft manuscript
├── missing_ablation_plan.md    # Ablation planning document
├── pipeline_redesign_report.md # Task interference analysis
└── presentation_report.md      # Summary of findings
```

### 2.2 Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Loads YAML configs into dictionaries |
| `paths.py` | Defines `ProjectPaths` for data_raw, data_processed, outputs |
| `data.py` | Loads AfriQA and MasakhaNER from HuggingFace; normalizes answers; exports to seq2seq JSONL format; NER tag linearization (`PER: X, LOC: Y`) |
| `train.py` | Tokenizes input/target pairs; builds `Seq2SeqTrainer` with `DataCollatorForSeq2Seq` |
| `eval.py` | Computes Exact Match, Token F1, LaBSE semantic similarity; handles `<extra_id>` cleaning, NER tag stripping, batch prediction generation with beam search |
| `logging_utils.py` | Standard Python logger with file + console output |

### 2.3 Data Flow

```
HuggingFace Hub (afriqa-gold-passages, masakhaner2)
  ↓  00_download_and_subset.py
data/processed/ (afriqa_swa_hau_yor, masakhaner2_swa_hau_yor)
  ↓  01_prepare_qa_data.py / 01b_prepare_multitask_data.py
data/processed/qa_seq2seq_swa_hau_yor{_multitask}/ (train/val/test.jsonl)
  ↓  02_train_baseline_qa.py / 03_train_multitask_qa.py
outputs/predictions/*.jsonl
  ↓  04_eval_predictions.py
outputs/metrics/*.json + *.csv
```

### 2.4 Key Design Decisions

1. **Seq2Seq framing**: Both QA and NER are framed as text-to-text generation (following T5 paradigm). QA input: `"question: <Q> context: <C>"` → target: `"<answer>"`. NER input: `"extract entities: <tokens>"` → target: `"PER: X, LOC: Y"`.
2. **Multitask data preparation**: QA examples are upsampled 20x before interleaving with NER examples to address the severe data imbalance (470 QA vs 19,185 NER train examples).
3. **Sequential training option**: NER Phase 1 → QA Phase 2 to prevent format leakage.
4. **LoRA integration**: Via HuggingFace PEFT, rank-16, targeting q/v attention projections.
5. **Matched-volume ablation**: QA-only training with 20x upsampling to control for training volume confound.

---

## 3. Current Experimental Pipeline

### 3.1 Completed Experiments (7 configurations with saved metrics)

| ID | Model | Training Condition | NER | QA Upsample | LoRA |
|----|-------|-------------------|-----|-------------|------|
| `baseline_mt5` | mT5-base | QA-only (1x) | No | No | No |
| `matchedqa_mt5` | mT5-base | QA-only (20x) | No | Yes | No |
| `matchedqa_byt5` | ByT5-base | QA-only (20x) | No | Yes | No |
| `multitask_mt5` | mT5-base | Sequential NER→QA | Yes | Yes | No |
| `multitask_byt5` | ByT5-base | Sequential NER→QA | Yes | Yes | No |
| `multitask_mt5_lora` | mT5-base + LoRA | Multitask | Yes | Yes | Yes |
| `multitask_byt5_lora` | ByT5-base + LoRA | Multitask | Yes | Yes | Yes |

### 3.2 Results Summary (n=848 AfriQA test examples)

| Configuration | EM | F1 | Semantic Sim |
|--------------|----:|----:|----:|
| Baseline mT5 (1x) | 0.032 | 0.071 | 0.293 |
| **ByT5 Matched-Volume QA (20x)** | **0.219** | **0.292** | **0.513** |
| mT5 Multitask (NER→QA) | 0.210 | 0.262 | 0.469 |
| mT5 Matched-Volume QA (20x) | 0.197 | 0.236 | 0.439 |
| ByT5 Multitask (NER→QA) | 0.192 | 0.272 | 0.476 |
| mT5 LoRA Multitask | 0.164 | 0.200 | 0.398 |
| ByT5 LoRA Multitask | 0.045 | 0.094 | 0.308 |

### 3.3 Evaluation Metrics

- **Exact Match (EM)**: Normalized string equality after lowercasing, punctuation removal, whitespace collapse.
- **Token F1**: SQuAD-style set-based token overlap (precision × recall harmonic mean).
- **Semantic Answer Similarity (SAS+)**: LaBSE cosine similarity between predicted and gold answers.

### 3.4 Missing Elements in the Pipeline

- ❌ No translation pipeline baseline comparison in the results (script `02b_eval_translation_pipeline.py` exists but no results are stored).
- ❌ No ByT5 low-resource baseline (1x QA-only for ByT5).
- ❌ No statistical significance testing.
- ❌ No training loss curves or convergence analysis saved.
- ❌ No cross-validation or multiple seeds.
- ❌ No per-language breakdown analysis in the paper draft beyond the top model.
- ❌ No NER-specific evaluation (entity F1 on MasakhaNER test set).
- ❌ No error analysis or qualitative examples in the outputs.
- ❌ No figures generated.

---

## 4. Summary of Every Research Paper

### 4.1 AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages

| Aspect | Detail |
|--------|--------|
| **Research Problem** | No QA benchmark exists for African languages; existing XOR QA benchmarks cover few African languages |
| **Main Contribution** | AfriQA dataset: 12,239 QA pairs across 10 African languages with gold passage annotations in both source and pivot languages |
| **Methodology** | 4-stage annotation pipeline: (1) question elicitation from Wikipedia prompts, (2) translation to pivot language, (3) answer labeling on retrieved passages, (4) answer translation back to source. Three evaluation tasks: XOR-Retrieve, XOR-PivotLanguageSpan, XOR-Full |
| **Datasets** | AfriQA (10 languages: Bemba, Fon, Hausa, Igbo, Kinyarwanda, Swahili, Twi, Wolof, Yoruba, Zulu) |
| **Models** | BM25, mDPR (multilingual BERT), AfroXLMR (extractive QA), mT5-base (generative QA), NLLB-200, Google Translate, MAFAND M2M-100 |
| **Evaluation Metrics** | Recall@k for retrieval, Exact Match (EM) and F1 for QA, BLEU for translation |
| **Baselines** | Human translation + BM25/mDPR retrieval + extractive/generative reader; NLLB translation; Google Translate |
| **Ablation Studies** | Translation system comparison (Human vs GMT vs NLLB vs M2M-100); retrieval system comparison (BM25 vs mDPR vs hybrid); gold passage vs retrieved passage QA |
| **Key Findings** | (1) Translation quality critically impacts downstream QA; (2) Human translation achieves best results but is expensive; (3) mDPR outperforms BM25 for most languages; (4) Very low-resource languages (Fon, Wolof, Bemba) have poorest performance |
| **Limitations** | (1) Heavy reliance on translation pipelines; (2) Wikipedia-centric questions; (3) Limited to 10 languages; (4) Gold passages assume access to relevant documents |

### 4.2 The Bitter Lesson (Rich Sutton, 2019)

| Aspect | Detail |
|--------|--------|
| **Research Problem** | Historical pattern in AI research where human-knowledge-based methods are consistently superseded by computation-scaling methods |
| **Main Contribution** | Articulation of the "Bitter Lesson": general methods leveraging computation (search + learning) consistently outperform domain-specific human knowledge engineering |
| **Methodology** | Historical survey and argumentation across chess, Go, speech recognition, computer vision |
| **Key Findings** | (1) Building in human knowledge helps short-term but plateaus; (2) Scaling computation via search and learning wins long-term; (3) The two most scalable techniques are search and learning; (4) We should build meta-methods that discover, not embed what we've discovered |
| **Limitations** | Position paper without quantitative evidence; does not address when domain knowledge *is* necessary (e.g., low-resource settings) |

### 4.3 Deep Double Descent: Where Bigger Models and More Data Hurt (Nakkiran et al., 2019)

| Aspect | Detail |
|--------|--------|
| **Research Problem** | Understanding the non-monotonic relationship between model complexity and test performance, where bigger models and more data can hurt |
| **Main Contribution** | Generalized double descent hypothesis: model-wise, epoch-wise, and sample-wise double descent unified under "Effective Model Complexity" (EMC) |
| **Methodology** | Systematic empirical study varying model size, training duration, number of train samples, and label noise across architectures |
| **Datasets** | CIFAR-10, CIFAR-100, IWSLT'14 de-en |
| **Models** | ResNet-18 (varying widths), 5-layer CNNs, 6-layer Transformers |
| **Evaluation Metrics** | Test error rate, train error |
| **Ablation Studies** | (1) Model size variation; (2) Training epoch variation; (3) Dataset size variation; (4) Label noise variation (0%–20%); (5) With/without data augmentation; (6) With/without early stopping |
| **Key Findings** | (1) Double descent occurs in model-wise, epoch-wise, and sample-wise dimensions; (2) Peak test error occurs at the interpolation threshold (EMC ≈ n); (3) More data can hurt in the critical regime; (4) Early stopping can mitigate but not always eliminate double descent; (5) Label noise amplifies the effect |
| **Limitations** | (1) No formal characterization of "sufficiently smaller/larger" relative to EMC; (2) ε parameter chosen heuristically; (3) Limited to classification tasks |

### 4.4 Eliminating Translation Pipelines in Multilingual QA for African Languages (Project Proposal)

| Aspect | Detail |
|--------|--------|
| **Research Problem** | Translation-based QA pipelines degrade for low-resource African languages |
| **Main Contribution** | Proposal for entity-aware multitask QA combining AfriQA + MasakhaNER with SAS+ evaluation |
| **Methodology** | Multi-task learning combining QA and NER; proposed 5 conditions (Baseline, Multi-task, Intermediate, Adapters, Family transfer) |
| **Datasets** | AfriQA gold-passages, MasakhaNER 2.0 |
| **Models** | mT5 with multi-task architecture, NER head |
| **Evaluation Metrics** | EM, F1, SAS+, Entity-F1 (per entity type) |
| **Key Findings** | Hypothetical results table only; this is a proposal, not a completed study |
| **Limitations** | Originally proposed 10 languages, 5 conditions — implementation covers only 3 languages and different conditions than proposed |

---

## 5. Comparison Between Our Work and Existing Literature

### 5.1 Ideas Already Implemented

| From Literature | Implementation Status |
|----------------|----------------------|
| AfriQA dataset as primary QA benchmark | ✅ Fully implemented (gold-passages subset, 3 of 10 languages) |
| mT5-base as generative QA model | ✅ Fully implemented |
| EM and F1 evaluation metrics | ✅ Fully implemented |
| Translation-pipeline QA baseline | ⚠️ Script exists (`02b_eval_translation_pipeline.py`) but **no results saved** |
| MasakhaNER 2.0 for entity-aware supervision | ✅ Fully integrated into multitask pipeline |
| The Bitter Lesson: scaling computation over knowledge engineering | ✅ Core finding validated — ByT5 + exposure scaling outperforms complex multitask |
| Deep Double Descent: training dynamics awareness | ⚠️ Implicit — the project discovered exposure scaling matters, but does not explicitly analyze double descent phenomena |
| SAS+ semantic evaluation (Ijebu et al. via proposal) | ✅ LaBSE-based semantic similarity implemented |
| LoRA parameter-efficient fine-tuning | ✅ Implemented and evaluated |
| Byte-level tokenization (ByT5) | ✅ Fully supported as alternative architecture |
| Sequential NER→QA training | ✅ Implemented to solve format leakage |
| Matched-volume ablation control | ✅ Critical experiment implemented and run |

### 5.2 Experiments That Are Missing

| Missing Experiment | Evidence of Need | Feasibility |
|-------------------|------------------|-------------|
| **ByT5 low-resource baseline (1x)** | Cannot compute ByT5 exposure delta without it | 🟢 Trivial |
| **Translation pipeline baseline** | Paper claims to "eliminate" these — must compare | 🟡 Script exists, needs to be run |
| **Statistical significance testing** | Required by all NLP venues; currently single-seed | 🟡 Medium — 3-5 seeds |
| **NER task evaluation** | Multitask models trained on NER but never evaluated on NER test set | 🟢 Easy |
| **Error analysis / qualitative examples** | Standard for QA papers | 🟡 Medium |
| **More languages (beyond swa/hau/yor)** | AfriQA covers 10 languages; 3/10 is narrow | 🔴 High effort |
| **Comparison with AfriQA's published baselines** | Must contextualize against AfriQA's reported numbers | 🟢 Easy — look up numbers |
| **Per-language delta analysis for all models** | Only top model has language breakdown in draft | 🟢 Easy — data in JSON files |
| **Training curve / convergence analysis** | Double descent paper is referenced but no convergence analysis done | 🟡 Medium |
| **Larger model sizes** | Bitter Lesson argues for scaling; only base-size tested | 🔴 Expensive |
| **Simultaneous multitask (joint training)** | Only sequential reported; code supports joint | 🟢 Easy |
| **Cross-lingual transfer analysis** | How does model perform on unseen languages? | 🔴 High effort |
| **Confidence intervals / variance** | Standard scientific rigor | 🟡 Multiple runs needed |

---

## 6. Experiments Already Covered

The repository covers a reasonably complete set of controlled experiments for its core claims:

1. ✅ **Baseline establishment**: mT5-base QA-only on raw 470 examples → EM 0.032
2. ✅ **Exposure scaling ablation**: 20x QA upsampling without NER → mT5 EM 0.197, ByT5 EM 0.219
3. ✅ **Multitask with entity supervision**: Sequential NER→QA → mT5 EM 0.210, ByT5 EM 0.192
4. ✅ **Architecture comparison**: mT5 (subword) vs ByT5 (byte-level) across all conditions
5. ✅ **Parameter efficiency**: LoRA (rank-16) vs full fine-tuning
6. ✅ **Delta analysis**: Decomposition of gains into exposure vs task components
7. ✅ **Multi-metric evaluation**: EM, F1, and semantic similarity

---

## 7. Missing Experiments

### 7.1 Critical Gaps (Would Be Flagged by Reviewers)

| # | Missing Experiment | Impact |
|---|-------------------|--------|
| 1 | **No ByT5 low-resource baseline** | Cannot fully compute ByT5 exposure/task deltas |
| 2 | **No translation pipeline comparison** | Paper title claims to "eliminate" translation pipelines but never demonstrates what they achieve |
| 3 | **No statistical significance** | Single-seed results; all deltas could be noise |
| 4 | **No comparison to AfriQA's published baselines** | Must compare against source benchmark numbers |
| 5 | **No NER evaluation** | Claims multitask teaches entity awareness but never measures NER performance |

### 7.2 Important But Not Critical

| # | Missing Experiment | Impact |
|---|-------------------|--------|
| 6 | Joint (non-sequential) multitask comparison | Only tested sequential |
| 7 | Per-language delta analysis for all models | Only shown for top model |
| 8 | Error analysis / qualitative examples | Needed for understanding failure modes |
| 9 | Training convergence curves | Would validate exposure-scaling claim quantitatively |

### 7.3 Nice to Have

| # | Missing Experiment | Impact |
|---|-------------------|--------|
| 10 | More languages (4-10 instead of 3) | Broader generalizability |
| 11 | Larger model sizes | Scaling analysis |
| 12 | Different LoRA ranks (4, 8, 32, 64) | Understanding adapter capacity |
| 13 | Context length ablation | How much passage context matters |

---

## 8. Low-Effort Improvements

**Can be done with the current codebase in < 1 day of compute + analysis:**

1. **Run ByT5 low-resource baseline**: Use `matchedqa_byt5.yaml` config but set `matched_qa_upsample_factor: 1` and run `02_train_baseline_qa.py`. Fills the most critical gap.

2. **Extract per-language deltas for all models**: The JSON files in `outputs_colab/metrics/` already contain per-language breakdowns for all 7 configurations. Just need analysis code.

3. **Compare against AfriQA published baselines**: The AfriQA paper reports EM/F1 for mT5-base on gold passages (SQuAD-trained, zero-shot). Cite and compare directly.

4. **NER evaluation on multitask predictions**: Evaluate multitask model predictions that have `lang=unknown` (NER examples) using entity-level metrics. Test predictions already exist.

5. **Generate result comparison tables and figures**: Currently no figures exist in `outputs/figures/`. Bar charts comparing all 7 configurations would strengthen the paper.

6. **Add a joint multitask run**: Remove the `--sequential` flag from `03_train_multitask_qa.py` — the code already supports this mode.

---

## 9. Medium-Effort Improvements

**Require 1-3 days of compute + analysis:**

1. **Run the translation pipeline baseline**: `02b_eval_translation_pipeline.py` exists and uses NLLB-200 + FLAN-T5. Run it, evaluate with `04_eval_predictions.py`, and add to results table. **The most impactful missing experiment for the paper's framing.**

2. **Multiple seed runs (3-5 seeds)**: Modify the `project.seed` config value and re-run the top 3-4 configurations. Report mean ± std.

3. **Error analysis notebook**: Create a notebook that loads predictions, categorizes errors (partial match, wrong entity, empty prediction, NER format leakage), and produces qualitative examples.

4. **Training convergence analysis**: Re-run experiments with TensorBoard logging or parse existing log files for loss curves.

5. **Simultaneous vs sequential multitask comparison**: Run the multitask training without `--sequential` to compare joint vs sequential NER→QA.

---

## 10. High-Effort Improvements

**Require > 1 week of effort:**

1. **Expand to 7-10 AfriQA languages**: Requires adding more language configs (igb, kin, twi, wol, zul, bem, fon), downloading MasakhaNER for all, re-running all pipeline scripts.

2. **Scaling analysis with larger models**: mt5-large (1.2B params), mt5-xl (3.7B params). Requires A100-level GPUs.

3. **Retrieval integration**: Replace gold passages with retrieved passages using BM25/mDPR to match AfriQA's XOR-Full setting.

4. **Entity-type conditioned evaluation**: Tag AfriQA questions by entity type (PER, LOC, ORG, DATE), then measure if multitask NER specifically helps entity-heavy questions.

5. **Cross-lingual zero-shot transfer**: Train on swa+hau, evaluate on yor (held-out language).

---

## 11. Overall Assessment of Publishability

### 11.1 Strengths

| Strength | Evidence |
|----------|----------|
| **Novel finding about exposure vs entity-awareness** | The delta analysis cleanly separates volume effects from NER effects — genuine contribution |
| **Architecture-dependent insight** | mT5 vs ByT5 divergence regarding NER supervision is novel and well-supported |
| **Proper ablation design** | Matched-volume control shows methodological rigor and self-correction |
| **Clean, reproducible codebase** | Well-structured modular code with YAML configs, bash runners, Colab notebooks |
| **Practical relevance** | Direct QA without translation is genuinely valuable for deployment in African language settings |

### 11.2 Weaknesses

| Weakness | Severity |
|----------|----------|
| **No translation pipeline comparison** | 🔴 Critical — title promises to "eliminate" them but never compares |
| **Only 3 of 10 AfriQA languages** | 🔴 Critical — limits generalizability |
| **No statistical significance** | 🔴 Critical — mT5 NER delta (+0.013 EM) may be noise |
| **Absolute performance is low** | 🟡 Major — best EM is 21.9% |
| **No ByT5 low-resource baseline** | 🟡 Major — delta analysis incomplete |
| **Missing NER evaluation** | 🟡 Major — claims entity-awareness without measuring it |
| **No error analysis** | 🟡 Major — standard for QA papers |
| **Limited related work** | 🟡 Major — paper draft has no Related Work section |
| **No figures** | 🟡 Minor |
| **Paper draft incomplete** | 🟡 Minor — no Abstract, Related Work, or References |

### 11.3 Publishability Verdict

**Current state: NOT ready for publication at a top venue (ACL, EMNLP, NAACL).** Could be a workshop paper (e.g., AfricaNLP, MasakhaNLP) with targeted improvements.

**Rationale:**
- The core scientific finding (exposure scaling > entity supervision, with architectural dependence) is novel and interesting.
- However, the experimental evidence is insufficient:
  - No comparison against the baseline the paper claims to replace (translation pipeline).
  - Only 3 languages is too narrow for a "multilingual African languages" claim.
  - Single-seed results with thin margins make the NER delta for mT5 (+0.013 EM) statistically unreliable.
  
**For a workshop paper (AfricaNLP Workshop, MasakhaNLP):** Current results with 3-5 targeted fixes (translation baseline, ByT5 baseline, significance testing, related work) could be sufficient.

**For a main conference paper:** Would need 7+ languages, multi-seed significance testing, translation pipeline comparison, error analysis, and stronger contextualization against AfriQA baselines.

---

## 12. Recommended Next Steps (Analysis Only)

### Priority 1: Minimum Fixes for Workshop Submission

1. **Run ByT5 low-resource baseline** — Fills the delta analysis gap.
2. **Run translation pipeline baseline** — Validates the paper's core framing.
3. **Add AfriQA published numbers to comparison table** — Contextualize results.
4. **Run 3 seeds for top-3 configurations** — Report mean ± std.
5. **Write Related Work and Abstract** — Essential missing sections.

### Priority 2: Strengthening for Conference Submission

6. **Expand to 5-7 languages** — Add Igbo, Kinyarwanda, Zulu.
7. **Conduct error analysis** — Categorize prediction errors by type.
8. **Generate figures** — Bar charts, delta visualizations, heatmaps.
9. **Evaluate NER performance** of multitask models on MasakhaNER test set.
10. **Compare joint vs sequential multitask** — Code supports both modes.

### Priority 3: Extended Analysis

11. **Training convergence curves** linking to Deep Double Descent insights.
12. **LoRA rank sweep** (4, 8, 16, 32) to understand adapter capacity.
13. **Entity-type conditioned evaluation** to test whether NER specifically helps entity-centric questions.

---

> **⚠️ SINGLE MOST IMPACTFUL ACTION**: Run the translation pipeline baseline (#2). The paper's entire framing — "eliminating translation pipelines" — currently has zero empirical support because the translation pipeline was never evaluated on the same test data.

> **⚠️ STATISTICAL WARNING**: The mT5 NER delta (+0.013 EM, +0.026 F1) is dangerously thin. Without significance testing, a reviewer could dismiss the entire entity-awareness finding for mT5. The ByT5 finding (negative NER delta) is stronger but still needs variance estimates.
