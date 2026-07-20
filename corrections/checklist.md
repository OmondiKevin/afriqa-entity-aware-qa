# Corrections Checklist
## Quick-Pick Reference for Fixing Paper Gaps

> Use this file to select which corrections to tackle. Each item references the full analysis in `comprehensive_analysis_report.md`.

---

## 🔴 Critical (Must Fix Before Any Submission)

- [ ] **C1 — ByT5 Low-Resource Baseline**
  - Run `02_train_baseline_qa.py` with ByT5 config, `upsample_factor: 1`
  - Effort: ~2-4 hours GPU
  - Why: ByT5 delta analysis is incomplete without this

- [ ] **C2 — Translation Pipeline Comparison**
  - Run `scripts/02b_eval_translation_pipeline.py` → then `04_eval_predictions.py`
  - Effort: ~2-4 hours GPU
  - Why: Paper title claims to "eliminate" translation pipelines but never shows what they score

- [ ] **C3 — Statistical Significance (Multi-Seed)**
  - Re-run top 3-4 configs with seeds 42, 123, 456 (minimum)
  - Effort: 3× current compute
  - Why: mT5 NER delta (+0.013 EM) may be noise without variance

- [ ] **C4 — AfriQA Published Baseline Numbers**
  - Look up mT5-base gold-passage results from AfriQA paper Table 5/6 for swa/hau/yor
  - Effort: 30 minutes (no compute)
  - Why: Must contextualize against source benchmark

- [ ] **C5 — NER Evaluation**
  - Evaluate multitask predictions on NER test subset (lang=unknown rows)
  - Effort: 1-2 hours
  - Why: Claims "entity-aware" but never measures NER accuracy

---

## 🟡 Important (Needed for Strong Workshop Paper)

- [ ] **I1 — Per-Language Deltas for All Models**
  - Parse all 7 JSON files in `outputs_colab/metrics/`, compute deltas
  - Effort: 1-2 hours scripting
  - Data already exists

- [ ] **I2 — Joint vs Sequential Multitask**
  - Run `03_train_multitask_qa.py` WITHOUT `--sequential` flag
  - Effort: ~4 hours GPU
  - Code already supports this

- [ ] **I3 — Error Analysis**
  - Load predictions, categorize: partial match, wrong entity, empty, format leakage
  - Effort: 1 day
  - Create a notebook with qualitative examples

- [ ] **I4 — Figures and Visualizations**
  - Bar charts: all 7 configs × 3 metrics
  - Delta decomposition chart (exposure vs task)
  - Per-language heatmap
  - Effort: half day

- [ ] **I5 — Training Convergence Curves**
  - Parse log files or re-run with TensorBoard
  - Effort: half day
  - Links to Deep Double Descent reference

- [ ] **I6 — Paper Draft: Related Work + Abstract**
  - Write missing sections
  - Cite: AfriQA, MasakhaNER 2.0, XOR QA, Ijebu et al., Bitter Lesson
  - Effort: 1 day writing

---

## 🟢 Nice to Have (Conference-Level)

- [ ] **N1 — Expand to 5-7 Languages**
  - Add: igb (Igbo), kin (Kinyarwanda), zul (Zulu)
  - Effort: 1-2 weeks (full re-run of pipeline)

- [ ] **N2 — Larger Model Sizes**
  - mt5-large (1.2B), requires A100 GPU
  - Effort: 1 week

- [ ] **N3 — LoRA Rank Sweep**
  - Test ranks: 4, 8, 16, 32, 64
  - Effort: 2-3 days GPU

- [ ] **N4 — Entity-Type Conditioned Evaluation**
  - Tag AfriQA questions by entity type, measure per-type impact
  - Effort: 3-5 days

- [ ] **N5 — Cross-Lingual Zero-Shot Transfer**
  - Train on swa+hau, test on yor
  - Effort: 2-3 days

- [ ] **N6 — Retrieval Integration**
  - Replace gold passages with BM25/mDPR retrieved passages
  - Effort: 1-2 weeks (new infrastructure)

---

## Quick Decision Matrix

| Targeting... | Do these items |
|-------------|---------------|
| **AfricaNLP Workshop** | C1 + C2 + C4 + I4 + I6 |
| **MasakhaNLP Workshop** | C1 + C2 + C3 + C4 + C5 + I1 + I4 + I6 |
| **EMNLP/ACL Main** | All Critical + All Important + N1 + N4 |
| **Maximum Impact / Minimum Effort** | C2 (translation baseline) + C4 (cite AfriQA numbers) |
