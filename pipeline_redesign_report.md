# Training Pipeline Redesign: Addressing Task Interference in Multitask QA

## Executive Summary

Recent Colab experiments with our `mt5-base` model revealed severe performance degradation in QA tasks when training jointly with Named Entity Recognition (NER). The multitask model suffered a **~72.4% drop** in Exact Match (EM) score, and the LoRA variant suffered a **~96.4% drop** compared to our QA-only baseline. 

This report outlines the root causes of this degradation—classified broadly as **Task Interference**—and details the concrete technical steps we are taking in our new implementation plan to resolve them.

---

## 1. Root Cause Analysis (The "Why")

We identified two primary factors driving the QA performance collapse:

### 1.1 Data Imbalance
Our training set contained a massive disparity between tasks: 19,185 NER examples compared to only 470 QA examples (a ~41:1 ratio). As a result, the optimizer's gradient updates were completely dominated by the NER task. The model essentially optimized for NER at the expense of QA, ignoring the QA task dynamics.

### 1.2 Format Leakage (Format Conflict)
While the multitask model successfully learned to extract correct entities (the "who" and "where"), it lost the ability to output clean text for QA. Instead of outputting a plain answer (e.g., `Israel`), the model heavily leaked NER tag formats into QA predictions (e.g., `LOC: Israel`). Because QA metrics rely on Exact Match against raw text, this format leakage caused logically correct extractions to be marked as complete failures.

> **Example of Format Leakage:**
> * **Question:** Who is the PM of Israel?
> * **Gold Answer:** Naftali Bennett
> * **Model Prediction:** `PER: Naftali Bennett, PER: Naftālī Beneṭ`

---

## 2. Redesign Plan (The "How")

To directly mitigate these issues, we have pivoted our pipeline to incorporate the following strategies:

### 2.1 Dynamic Data Rebalancing
We are introducing a `qa_upsample_factor` (default: 20x) in the data preparation phase. By aggressively upsampling the QA data before merging it with NER, we bring the QA-to-NER ratio closer to a balanced 1:2. This stops the NER signal from washing out QA during training updates.

### 2.2 Shift to Sequential Training
To completely solve format leakage, we are abandoning simultaneous joint training. Instead, a new `--sequential` pipeline will:
1. **Phase 1:** Train strictly on the NER dataset to teach the model robust entity recognition capabilities.
2. **Phase 2:** Load this NER-trained checkpoint and fine-tune it **exclusively on the QA dataset**.

By training sequentially, the model benefits from the rich entity representations learned in Phase 1, but "unlearns" the strict `PER:` / `LOC:` formatting constraints while focusing entirely on QA in Phase 2.

### 2.3 Extended Architecture Support (ByT5)
We have added infrastructure support to easily swap models via a centralized `model.base` config, specifically adding **ByT5**. Unlike `mT5` which relies on SentencePiece tokenization, ByT5 operates directly at the character level. This gives us a powerful alternative architecture that is often more resilient to formatting anomalies and the complex morphology innate to agglutinative African languages.

### 2.4 Resilient Evaluation Mechanisms
To guarantee accurate metrics moving forward, we are adding tag-stripping utilities (e.g., `strip_ner_tags`) directly into the evaluation pipeline. If the model accidentally leaks prefixes during inference, the evaluator will strip them prior to scoring, ensuring we measure true entity extraction accuracy rather than minor formatting variations.

---

## Conclusion
The new pipeline correctly diagnoses the empirical failures we observed and transitions us from a naive joint-batch approach to a controlled, sequential training regimen. With rebalanced data, robust evaluation, and targeted tokenization enhancements, we are well-positioned to achieve high-performance, entity-aware QA.
