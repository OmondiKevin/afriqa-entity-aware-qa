# Executive Summary
**Title Concept:** Eliminating Translation Pipelines in Multilingual Question Answering for African Languages via Exposure Scaling

**Core Narrative:**
Cross-lingual QA systems for African languages often rely on zero-shot translation pipelines, which strip cultural context and compound errors. Previous attempts at direct multilingual QA struggled because they used standard low-resource fine-tuning (1x exposure), resulting in poor performance (EM=0.025 for ByT5).

This study provides strong empirical evidence that direct multilingual QA natively outperforms translation pipelines (EM=0.116) when models undergo "Exposure Scaling"—upsampling the training data to drastically increase gradient steps (20x exposure). With matched volume, ByT5 achieves EM=0.219 on the AfriQA benchmark.

Crucially, we uncover a tokenization divergence:
- **Subword models (mT5)** struggle with agglutinative African morphology and statistically depend on explicit multitask entity-awareness (NER supervision) to improve answer boundaries (+0.013 Task Delta).
- **Byte-level models (ByT5)** naturally synthesize entities without explicit constraints. Forcing them into rigid multitask NER routing actively degrades their performance (-0.027 Task Delta). They achieve peak performance purely through exposure scaling.\n