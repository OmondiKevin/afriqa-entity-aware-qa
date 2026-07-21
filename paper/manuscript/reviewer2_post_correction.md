# Post-Correction Reviewer #2 Assessment
**Date:** 2026-07-22  
**Status of manuscript:** Corrected — awaiting ByT5 re-evaluation and multi-seed validation

---

## 1. Strongest Contribution

The strongest and most defensible contribution in this paper is the **mT5 exposure delta**:

> A model trained on 1× AfriQA data (mT5 EM = 0.032) is below a closed-book translation pipeline (EM = 0.116). The same model trained with 20× QA upsampling (EM = 0.197) surpasses it. The gap is closed entirely by gradient exposure — no architectural change, no multilingual transfer heuristics.

This finding:
- Uses only mT5 (so the generation-cap issue does not apply)
- Involves clean numbers from a real controlled experiment
- Has a concrete practical implication (naive fine-tuning on African QA fails not because the task is impossible, but because training volume was insufficient)
- Explains a common failure mode in the literature

This is the sentence that should appear in every abstract, and it should be the thesis of the paper.

---

## 2. Remaining Scientific Weaknesses

**W1 (BLOCKER):** All ByT5 metrics are under a 16-byte generation cap. Any claim that involves ByT5 absolute values — including "ByT5 QA-Upsampled is the best configuration" — cannot be stated confidently until corrected values are available. The current paper annotates these with "†" but the tables are still prominent.

**W2 (BLOCKER for causal claims):** Single seed. The mT5 task delta (+0.013 EM) is 8% of mT5 exposure delta (+0.165 EM). It is plausibly noise. The ByT5 task delta (−0.027† EM) is larger but also subject to the generation cap issue. No statistical claim is permissible.

**W3 (MAJOR):** The translation baseline is a closed-book parametric system (no passage). Comparing it to reading-comprehension fine-tuned models makes the comparison stronger for our side but less scientifically valid. A reviewer will demand a passage-reading translation baseline (e.g., translate passage + question → English → extractive QA → translate answer back).

**W4 (MAJOR):** Swahili zero-shot. The fact that Swahili was never in training is not mentioned prominently enough in the original draft. It has now been corrected (§4.1 and Table B1), but any reviewer reading the language-specific results will immediately notice that all Swahili numbers are zero-shot and cannot be compared to Hausa/Yoruba fine-tuning results.

**W5 (MODERATE):** LoRA collapse for ByT5 (EM = 0.045†, 23% retention) vs LoRA mT5 (EM = 0.164, 78% retention) is interesting but the paper has no ablation over rank, adapter placement, or total parameter count. The claim that "rank-16 is insufficient for byte-level multitask" is untested.

---

## 3. Likely Reviewer Objections

**Objection 1:** "You report ByT5 metrics with a known 16-byte cap. Why should we believe any ByT5 results?"

*Response in paper:* The paper now (§4.4, §8) explicitly states these are lower bounds, provides the truncation statistics, and identifies corrected evaluation as a required step. The paper correctly leads with the mT5 findings, which are not affected. However: if the corrected ByT5 metrics are dramatically higher, the task delta sign for ByT5 might change — which would undermine the tokenization-supervision interaction claim. This is a genuine scientific risk.

**Objection 2:** "Your translation baseline doesn't read the passage. This comparison doesn't mean anything."

*Response in paper:* The paper now (§3.2, Appendix F) discloses this clearly. The framing is: even with this disadvantaged baseline, proper exposure still matters. But the comparison cannot establish supremacy over a passage-reading translation pipeline. This should satisfy methodologically careful reviewers, but will make the paper's claims more modest.

**Objection 3:** "You have n=848 test examples, one seed, and three languages. How can you claim a 'tokenization-supervision interaction'?"

*Response in paper:* The paper now explicitly hedges ("consistent with a mechanism," "suggests," "directionally negative in 7/9 comparisons"). However, no p-value exists, and the current draft cannot say "statistically significant." This is the fundamental limitation of the study.

**Objection 4:** "Swahili zero-shot results should not appear in the same table as Hausa/Yoruba fine-tuning results without a clear visual separator."

*Response:* Tables now note SWA = zero-shot. A reviewer may still object to including them as peer comparisons. Consider separating Table C1 into fine-tuned (Hausa, Yoruba) and zero-shot (Swahili) panels.

**Objection 5:** "The NLLB+FLAN-T5 pipeline uses 600M + 250M parameters without passage context. Your 'direct' models have 580M parameters with passage context. This is not a controlled comparison."

*Response:* Correct and disclosed. We cannot fix this retroactively — this is the design of the experiment. The paper now makes this explicit.

---

## 4. Claims That Still Require Caution

| Claim | Risk Level | Status |
|---|---|---|
| "ByT5 QA-Upsampled is the top-performing configuration" | HIGH | Lower bound only. May not hold after correction. |
| "NER supervision consistently hurts ByT5" | HIGH | 7/9 comparisons, zero-shot exception, single seed. |
| "Exposure scaling is the primary driver" | MODERATE | Supported by mT5 evidence. Holds at single seed. |
| "LoRA collapses for ByT5" | MODERATE | Observed, not tested causally. |
| "Translation pipeline is an inferior approach" | MODERATE | Valid for closed-book comparison only. |
| "mT5 task delta is positive" | MODERATE | +0.013 EM is small; may be noise at single seed. |

---

## 5. Experiments Still Missing (Priority Order for Submission)

1. **ByT5 re-evaluation at corrected generation limits (32, 64, 128 bytes)** — Required before submission. Without this, ByT5 tables are incomplete.

2. **Multi-seed validation (seeds 42, 123, 456) for:** QA-Upsampled mT5, QA-Upsampled ByT5, Multitask mT5 — These three configurations carry the core claims. Multi-seed is necessary for any claim of statistical reliability.

3. **Swahili QA training data** — Acquire Swahili AfriQA training examples (they exist in the full AfriQA corpus but were absent in the gold-passage subset). This would remove the zero-shot exception and strengthen cross-language generalization claims.

4. **Passage-reading translation baseline** — Replace FLAN-T5 closed-book with an extractive reader on translated passage+question. This makes the comparison scientifically valid.

5. **LoRA rank ablation for ByT5** — Test rank 32, 64, full fine-tuning to understand where ByT5 LoRA collapses.

---

## 6. Submission Readiness

**Current status: NOT READY FOR SUBMISSION.**

### Blockers (must resolve before any submission):
- [x] ~~Learning rate asymmetry claim~~ — Corrected (both architectures used 1e-4)
- [x] ~~"Consistently across all languages" for ByT5 NER harm~~ — Corrected (7/9)
- [x] ~~Swahili zero-shot not disclosed~~ — Corrected (§4.1, Table B1)
- [x] ~~Translation pipeline described as "standard industrial"~~ — Corrected (closed-book disclosed)
- [x] ~~Max source length contradiction~~ — Resolved (runtime logs are authoritative)
- [ ] **ByT5 re-evaluation at corrected generation limit** — Required, not yet done
- [ ] **Per-language table B1 shows Swahili absent from training** — Disclosed, but data still absent
- [ ] **FLAN-T5 citation unverified** — Must verify Chung et al. 2022 before final submission

### Items ready for LaTeX conversion:
- mT5 tables (Table 1 mT5 rows, Table 2 mT5 rows, Table 3 mT5 rows)
- Appendix A hyperparameter table
- Appendix B dataset table (including the zero-shot disclosure)
- Appendix D training step analysis
- Appendix F translation pipeline description
- References (except FLAN-T5 and potentially NLLB)

### Items requiring update after ByT5 re-evaluation:
- Table 1 (all ByT5 rows)
- Table 2 (ByT5 delta rows)
- Table 3 (ByT5 per-language rows)
- Table C1, C2 (Appendix)
- All "†" annotations (should be removed)
- §5.1, §5.2, §5.3 prose about ByT5 values
- Abstract ByT5 numbers
- Conclusion ByT5 claims

---

## 7. Recommendation for the Next Action

**Step 1 (this week, highest priority):** Re-evaluate ByT5 checkpoints on Colab at gen_limits 32, 64, 128. Run `scripts/04_eval_predictions.py` with modified config. This is the single most critical step.

**Step 2 (concurrent):** Convert mT5 tables and sections to LaTeX while waiting for ByT5 correction. Use placeholders (†[CORRECTED VALUE]) in ByT5 cells.

**Step 3 (next sprint):** Run 3 seeds for QA-Upsampled mT5 and Multitask mT5 to get variance estimates on the two most important configurations.

**Step 4:** Verify FLAN-T5 citation (Chung et al. 2022). Confirm the arXiv ID 2210.11416 and author list.

**Step 5 (optional but strengthening):** Source a passage-reading translation baseline to replace the closed-book comparison.

**Target venue (suggested):** EMNLP Findings 2025 or ACL ARR (rolling). The paper is appropriate scope for Findings rather than main track given single-seed, three-language scope — unless multi-seed validation confirms the interaction finding.

