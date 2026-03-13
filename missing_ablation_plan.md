# Missing Ablation Recovery Plan for the Paper

## 1) Title

**Missing-ablation recovery plan for validating entity-aware QA claims in AfriQA experiments**

## 2) Current Problem

The current hypothesis ("entity-aware multitask supervision improves QA") is directionally supported, but not cleanly validated as a causal claim in the current experiment set.

From the current draft and run setup:

- Baseline QA-only (`baseline_mt5`) trains on 470 QA train examples.
- Multitask runs train on a much larger combined corpus (QA upsampled by `multitask.qa_upsample_factor=20` plus MasakhaNER), with reported multitask train size 28,585.
- Sequential multitask (`scripts/03_train_multitask_qa.py --sequential`) also changes training signal: Phase 1 learns NER tagging behavior, then Phase 2 fine-tunes on QA.

This introduces a **training-volume confound** and **training-signal confound** at the same time. The current comparison is therefore not an isolated test of entity-awareness.

## 3) Why This Matters

The current baseline-vs-multitask gap cannot isolate whether gains come from:

1. NER supervision (the intended mechanism), or
2. More optimization exposure (many more examples/repetitions/steps), or
3. Both.

A reviewer can reasonably challenge the claim by asking whether the observed gains are mostly due to additional QA exposure (or total training budget), rather than entity-aware supervision itself.

Without a matched QA-only control, the strongest defensible statement remains:
"the combined intervention (more exposure + multitask NER/QA structure) improves QA."

## 4) Missing Experiment(s)

### Required controlled ablation

Add a **QA-only matched-volume control** with no NER supervision:

- Train QA-only with matched upsampling/training budget used to compare against multitask.
- Keep architecture, optimizer family, LR schedule, early stopping policy, and eval pipeline aligned with current runs.
- Evaluate on the same AfriQA test split and same metrics.

### Scope

- **Minimum required:** `mT5-base` matched-volume QA-only ablation.
- **Recommended if practical and pipeline-consistent:** `ByT5-base` matched-volume QA-only ablation.

### Practical control definition

Primary control should match multitask QA exposure (upsampled QA) and ideally also match total optimizer-step budget to the multitask run being compared. If only one is feasible quickly, prioritize matched QA upsampling first and explicitly report any residual step mismatch.

## 5) Experimental Cases Table

| Experiment ID | Model | Training setup | NER used | QA upsampled | LoRA used | Purpose |
|---|---|---|---|---|---|---|
| `baseline_mt5` | `google/mt5-base` | QA-only baseline (`scripts/02_train_baseline_qa.py`) | No | No (1x) | No | Reference low-resource baseline |
| `multitask_mt5` | `google/mt5-base` | Sequential multitask NER->QA (`scripts/03_train_multitask_qa.py --sequential`) | Yes | Yes (factor from config, currently 20x) | No | Main proposed method |
| `multitask_byt5` | `google/byt5-base` | Sequential multitask NER->QA | Yes | Yes (same multitask prep policy) | No | Architecture comparison under multitask |
| `multitask_mt5_lora` | `google/mt5-base` + LoRA | Sequential multitask NER->QA with adapters | Yes | Yes | Yes | Parameter-efficient comparison (mT5) |
| `multitask_byt5_lora` | `google/byt5-base` + LoRA | Sequential multitask NER->QA with adapters | Yes | Yes | Yes | Parameter-efficient comparison (ByT5) |
| `matchedqa_mt5` (new) | `google/mt5-base` | QA-only matched-volume control (no NER phase) | No | Yes (matched to comparison target) | No | Isolate entity-awareness from exposure volume |
| `matchedqa_byt5` (new, optional) | `google/byt5-base` | QA-only matched-volume control (no NER phase) | No | Yes (matched policy) | No | Same control for ByT5 track |

## 6) Proposed Naming Conventions

To stay collision-free and consistent with current `outputs_colab/{predictions,metrics,logs}` structure and current stem-based metric generation in `scripts/04_eval_predictions.py`:

- Predictions:
  - `outputs/predictions/matchedqa_mt5_test.jsonl`
  - `outputs/predictions/matchedqa_byt5_test.jsonl` (if run)
- Metrics:
  - `outputs/metrics/matchedqa_mt5_test_metrics.json`
  - `outputs/metrics/matchedqa_mt5_test_metrics.csv`
  - `outputs/metrics/matchedqa_byt5_test_metrics.json` (if run)
  - `outputs/metrics/matchedqa_byt5_test_metrics.csv` (if run)
- Logs:
  - `outputs/logs/02_train_matchedqa_mt5.log` (or `02_train_baseline_qa.log` copied/renamed after run)
  - `outputs/logs/02_train_matchedqa_byt5.log` (if run)
  - `outputs/logs/04_eval_predictions_matchedqa_mt5.log`
  - `outputs/logs/04_eval_predictions_matchedqa_byt5.log` (if run)

Notes:

- Because these files are QA-only predictions, `--qa_only` is not required for evaluation.
- Use explicit `matchedqa_` prefix to avoid collisions with `baseline_` and `multitask_` artifacts.

## 7) Code Changes Likely Needed

Likely edit targets (after this planning document is approved):

- `configs/default.yaml` (or a dedicated derived config)  
  Add run paths and matched-volume controls (e.g., `matchedqa_*` output names, upsample/step controls).
- `scripts/02_train_baseline_qa.py`  
  Add a matched-volume mode for QA-only upsampling and/or step matching.
- `scripts/04_eval_predictions.py`  
  Probably no logic change needed; keep using `--pred_path` stem-based metric naming.
- `run_experiments_colab.ipynb`  
  Add a matched-volume QA-only run cell and metric collection row.
- `run_lora_experiments_colab.ipynb`  
  Optional: add matched QA-only controls only if that notebook remains the source of aggregate tables.
- Optional new notebook: `run_matched_ablation_colab.ipynb`  
  Cleaner if we want minimal disturbance to existing demo flow.

## 8) Recommended Implementation Strategy

**Recommended:** extend `scripts/02_train_baseline_qa.py` with a small matched-volume mode, and drive it via a dedicated config variant.

Why this is the cleanest fit:

- Reuses the existing QA-only training/eval path directly.
- Avoids duplicating trainer logic in a new script.
- Keeps ablation semantics explicit and reviewable in config.
- Minimizes disruption to current multitask and LoRA scripts.

Concrete pattern:

1. Add a config block for matched QA control (e.g., `ablation.matched_qa_upsample_factor`, optional `ablation.match_total_steps`).
2. Add a lightweight CLI flag (e.g., `--matched_volume`) in `02_train_baseline_qa.py`.
3. Write outputs with `matchedqa_*` names using existing run config wiring.

## 9) Minimum Publishable Fix

Smallest experiment set needed to rescue the main claim:

1. Run `matchedqa_mt5` (QA-only, matched-volume, no NER).
2. Evaluate and report side-by-side:
   - `baseline_mt5`
   - `matchedqa_mt5`
   - `multitask_mt5`

This is sufficient to test whether multitask gains persist after exposure-volume control for the main architecture used in baseline comparisons.

`matchedqa_byt5` is strongly recommended but not strictly required for the minimum publishable fix.

## 10) Success Criteria

Interpretation rules for the new ablation:

- If `multitask_mt5` > `matchedqa_mt5` by a meaningful margin across EM/F1/Semantic, this supports an entity-awareness effect beyond training volume.
- If `matchedqa_mt5` ~= `multitask_mt5`, current gains are likely dominated by exposure volume/training budget rather than NER supervision.
- If `matchedqa_mt5` > `baseline_mt5` but < `multitask_mt5`, then both exposure and entity-aware supervision likely contribute.
- If `matchedqa_mt5` collapses to baseline, implementation or budget matching is likely incorrect and should be audited before interpretation.

For reporting, include absolute metrics and deltas relative to both baseline and matched-volume control.

## 11) Recommended Next Steps

1. Freeze this plan and keep it as the experiment-control reference.
2. Implement matched-volume mode in QA baseline training path (minimal script change).
3. Add one notebook run block for `matchedqa_mt5` and metric export.
4. Execute and archive artifacts under `outputs_colab/{predictions,metrics,logs}` with `matchedqa_*` names.
5. Update result tables and claims only after the new control is complete.

