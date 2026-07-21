# Reviewer #2 Attack List
### Critical Vulnerabilities
1. **Missing Statistical Significance:** "The authors claim SOTA, but without multi-seed standard deviations (e.g., seed 42, 123, 456), a jump of a few EM points could be initialization noise."
2. **Unfair Translation Comparison:** "The authors compare a highly-upsampled (200 epoch) fine-tuned model against a zero-shot translation pipeline. This is an apples-to-oranges comparison in terms of compute."
3. **NER Evaluation Flaw:** "The Multitask ByT5 NER F1 is 0.001. Reviewers will argue the multitask training simply failed or was bugged, rather than proving a structural hypothesis about byte-level models."

### Important
1. **Limited Language Scope:** "Only 3 languages out of 2000+ in Africa."\n