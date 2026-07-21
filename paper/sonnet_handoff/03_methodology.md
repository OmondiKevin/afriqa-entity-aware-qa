# Methodology

**Goal:** Isolate the effects of gradient exposure (volume) from architectural morphology (tokenization) and multitask supervision (entity routing).

**Configurations Tested:**
1. **Baseline (1x):** Standard fine-tuning on the 470 QA examples.
2. **Translation Pipeline:** Google Translate inference to a high-resource pivot (English), followed by English QA, and translated back.
3. **Matched-Volume (20x):** Strict exposure ablation. QA examples upsampled 20x to simulate the exact number of gradient steps used in multitask training without the multitask data.
4. **Multitask (NER -> QA):** Sequential multitask learning using MasakhaNER 2.0 interleaved with AfriQA to force entity-awareness.
5. **LoRA:** Parameter-efficient variants of the Multitask configuration using rank-16 adapters.

**Metrics:** Exact Match (EM), Token F1, Semantic Similarity (LaBSE cosine correlation).\n