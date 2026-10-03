> Historical pre-training notes. Update: authorized engineering fits completed on 2026-09-16; see [current training status](../../docs/TRAINING_PIPELINE.md). Full population training remains pending.

# Feature preparation and training-code status

## Completed

- All 85 runs for patient 125 have extracted feature tables.
- 56,285 windows; 367 unique named feature columns.
- 82 runs have 367 features; 3 runs have 235 because EEG-device motion channels are absent.
- 17,171 eligible windows: 17,051 background and 120 positive.
- Feature files occupy about 156.6 MiB.
- 18 tests passed without fitting a model or feature selector.
- Full feature audit passed: row matching, names, numerical values, labels, historical support and positive-event timing.

The eligible count is lower than the previous preprocessing-only result because the new ECG features require 60 seconds of observed history. All positive windows still belong to one seizure, so there is no demonstrated population forecasting performance.

## Ready in code, not executed

- Training-fold missing-value/constant cleanup and correlation pruning.
- Median imputation, scaling, missing indicators and configurable ANOVA/mutual-information ranking.
- Comparison of logistic regression, random forest and histogram gradient boosting.
- Selection-size comparison inside grouped cross-validation.
- Subject-separated train/validation/test checks.
- Validation alarm threshold choice and fixed held-out evaluation.
- Saved model, selected feature names/groups, split/configuration snapshots, scores and predictions.

No fitted ranking, trained model, hyperparameter search or held-out test score exists. Running python -m scripts.train without --start-training performed only a readiness check and correctly reported that explicit subject partitions and more data are needed.

## Before training

Copy additional patient folders into data/raw. The config now includes all downloaded subjects. Run preprocessing and feature preparation to include those recordings. Then define disjoint train/validation/test subject lists with enough eligible seizures. Wait for explicit authorization before passing --start-training.

See docs/RESEARCH_DECISIONS.md for primary sources and methodological limitations. Within-run synchronization is assumed from the publisher-aligned export; beat detection is a tested heuristic, not independently validated against clinical R-peak annotations. Fitted feature selection does not establish a physical-electrode count.
