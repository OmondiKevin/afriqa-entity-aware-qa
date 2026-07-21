# Executive Summary
**Title Concept:** Eliminating Translation Pipelines in Multilingual Question Answering for African Languages via Exposure Scaling

**Core Narrative:**
Cross-lingual QA systems for African languages often rely on zero-shot translation pipelines, which strip cultural context and compound errors. Previous attempts at direct multilingual QA failed because they used standard low-resource fine-tuning (1x exposure), resulting in poor performance (EM=0.025 for ByT5).

This study proves that direct multilingual QA natively beats translation pipelines (EM=0.116) when models undergo "Exposure Scaling"—upsampling the training data to drastically increase gradient steps (20x exposure). With matched volume, ByT5 achieves SOTA (EM=0.219).

Crucially, we uncover a tokenization divergence:
- **Subword models (mT5)** struggle with African morphology and statistically depend on explicit multitask entity-awareness (NER supervision) to find answer boundaries (+0.013 Task Delta).
- **Byte-level models (ByT5)** naturally synthesize entities. Forcing them into rigid multitask NER routing actively corrupts their performance (-0.027 Task Delta). They achieve SOTA purely through exposure.\n