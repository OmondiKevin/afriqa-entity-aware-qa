# Error Analysis
NER tags (PER, LOC, ORG, DATE) were evaluated using a strict exact match boundary metric.
- Both mT5 and ByT5 struggle to natively output perfect NER tags when combined with QA.
- In Multitask ByT5, only 41 predictions were structurally parseable for NER, achieving F1=0.001. This corroborates the hypothesis that forcing byte-level models to output rigid intermediate tagging corrupts their generation flow.
- Qualitative categories to note for the paper: "Partial Boundary Match", "Entity Type Confusion", "Hallucinated Context".\n