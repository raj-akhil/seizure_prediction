# Modular training pipeline: implementation and smoke-test results

Validation extension: the suite now has 32 passing tests and training automatically writes research-informed metric reports. See [validation pipeline](VALIDATION_PIPELINE.md) for current outputs, metric definitions and limitations. The 21-test count below records the earlier training implementation milestone.

## Status

The training implementation now has a successful actual-fit engineering test. Full population training on real patients has not started.

- 21 automated tests passed.
- End-to-end synthetic EDF test: five separate synthetic subjects, all four modalities, 235 eligible windows.
- Train/validation/test: three subjects (141 windows), one (47), one (47).
- Three inner subject-grouped folds; fitted feature selection and model comparison used training subjects only.
- Logistic regression, random forest and histogram gradient boosting all fitted.
- Validation selected the alarm threshold before held-out test evaluation.
- Reloaded saved-model predictions matched saved predictions to numerical tolerance.
- Real-data compatibility: each candidate fitted and predicted on 240 patient-125 rows (60 positive, 180 background). No generalization metrics were reported for this in-sample check.

Synthetic signals contain deliberately constructed patterns and use a permissive smoke-test false-alarm budget. Scores in the synthetic artifacts are not seizure-prediction results. Synthetic fixtures and models live only in outputs/smoke_tests and never enter data/raw or production feature manifests.

## Organization

| Module | Responsibility |
|---|---|
| src/utils/config.py | Configuration loading and validation |
| src/data/bids_loader.py | Recording discovery |
| src/preprocessing/ | Signal windows, timing, quality and labels |
| src/pipeline/preparation.py | Preprocessing orchestration and progress manifests |
| src/features/ | Deterministic modality feature extraction |
| src/training/data.py | Feature tables, schema alignment and resource checks |
| src/training/splits.py | Subject partitions, class/event checks and grouped folds |
| src/training/registry.py | Candidate models and configurable search grids |
| src/models/selection.py | Training-fold cleanup, scaling and feature ranking |
| src/models/evaluation.py | Window and conditional event/alarm metrics |
| src/training/runner.py | Fitting, model selection, persistence and evaluation |
| src/models/training.py | Compatibility exports for existing imports |
| python -m scripts.preprocess / python -m scripts.prepare_features / python -m scripts.train | Small stage-specific command-line entry points |
| python -m tests.smoke.training | Isolated actual-fit end-to-end engineering test |

## Commands

```powershell
.venv/Scripts/python.exe -m scripts.preprocess
.venv/Scripts/python.exe -m scripts.prepare_features
.venv/Scripts/python.exe -m scripts.train
# The line above checks readiness without fitting.
# After real-data partitions are ready and training is authorized:
.venv/Scripts/python.exe -m scripts.train --start-training
# Repeat the isolated engineering smoke test:
.venv/Scripts/python.exe -m tests.smoke.training
```

The successful real fitting test exposed two implementation details now handled: constant missingness indicators are excluded from ANOVA ranking, and native numerical threads are bounded using training.native_threads (default 1). Histogram boosting's internal random-window early stopping remains disabled.

## Testing before all 125 patients are downloaded

A practical next engineering cohort is at least five real patients: three training patients for three grouped inner folds, one validation patient and one test patient. Choose recordings that provide both background and eligible pre-seizure periods. Current configuration requires at least two eligible seizure episodes per outer partition. Five patients are enough to exercise this validation structure, not establish robust population accuracy. More patients, seizure diversity and monitored background are needed for scientific evaluation.

The current patient 125 data has only one eligible seizure under the 30–40-minute definition, so a meaningful real train/validation/test experiment is not possible yet. Never split its neighbouring positive windows across partitions or rename copied windows as different patients.

Once patients are copied into data/raw, rerun preparation, then explicitly assign disjoint subject lists in config/config.yaml. Existing compatible caches are reused. The pipeline rejects stale/incomplete feature manifests and overlapping patient partitions.

## Remaining scientific limits

Publisher-relative synchronization remains an assumption. ECG beat detection is heuristic. Evaluation is conditional on eligible windows; it does not establish full-day clinical false-alarm performance. Seizure-type target filtering exists, but automatic seizure-type-balanced subject allocation is not implemented. Smoke tests do not resolve these scientific limitations.
