# Reviewer #2 Attack List
### Critical Vulnerabilities
1. **Missing Statistical Significance:** "The authors claim a new baseline, but without multi-seed standard deviations (e.g., seed 42, 123, 456), a jump of a few EM points could be initialization noise."
   - *Sonnet Action:* Do not claim "state-of-the-art" definitively. Use "strong baseline". Add an explicit paragraph in the Limitations section acknowledging that these results represent a single seed and require variance testing for absolute statistical certainty.
2. **Translation API Unfairness:** "Comparing a highly-upsampled (200 epoch) fine-tuned model against a generic Google Translate API is an unfair baseline. What about NLLB-200 or an open-source NMT SOTA?"
   - *Sonnet Action:* Acknowledge in Limitations that the translation pipeline relies on commercial black-box APIs, which may not represent the ceiling of translation-based QA, though it represents standard industry practice.
3. **NER Evaluation Flaw (F1=0.001):** "The Multitask ByT5 NER F1 is 0.001. This suggests the multitask training simply collapsed or failed, rather than proving a structural hypothesis about byte-level models."
   - *Sonnet Action:* Discuss this explicitly. Acknowledge that the extreme failure of the NER generation in ByT5 *is* the point—the byte-level architecture fundamentally rejects being forced to output rigid sequence-to-sequence structural tags while simultaneously doing QA, whereas mT5 (F1=0.070) can handle the dual structure slightly better.

### Important
1. **Limited Language Scope:** "Only 3 languages out of 2000+ in Africa."
   - *Sonnet Action:* Keep claims strictly constrained to the tested typologies (Agglutinative Swahili, tonal isolating Yoruba, etc.).\n