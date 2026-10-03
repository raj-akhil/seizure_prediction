# Seizure forecasting validation

## Scientific choices

There is no single universally accepted metric or fully standardized seizure-forecasting evaluation protocol. Report discrimination, probability quality and event-warning performance together. This implementation evaluates **forecasting**, not seizure detection.

Research informing this implementation:

- [Saito and Rehmsmeier, 2015](https://doi.org/10.1371/journal.pone.0118432): precision–recall evaluation is particularly informative for imbalanced binary datasets. We retain **average precision (AP)** as the training-only model selection metric. AP uses a step-weighted PR integral; it is not trapezoidal PR-AUC. The evaluation prevalence is reported alongside it.
- [Comparison between epileptic seizure prediction and forecasting based on machine learning, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10920642/): evaluates seizure sensitivity, time in warning, Brier score and Brier skill score, and describes prediction horizon/occurrence-period conventions. These motivate the event and probabilistic metrics here; definitions below make our particular implementation explicit.
- [SeFEF: A Seizure Forecasting Evaluation Framework, 2025 preprint](https://arxiv.org/abs/2510.11275): combines deterministic and probabilistic evaluation and emphasizes protocol documentation and calibration. It describes an effort toward standardization, not an established universal consensus. Our population patient-held-out split is distinct from its individualized forecasting workflow.

## Metrics and interpretation

| Metric | Interpretation |
|---|---|
| Event sensitivity | Unique warned eligible seizures / unique eligible seizures; higher is better |
| False alarms per eligible hour/day | Unmatched alarm episodes / eligible monitoring exposure; lower is better |
| Time in warning | Fraction of eligible time support covered by a warning; lower burden at comparable sensitivity is better |
| Alarm precision | Alarm episodes matching at least one target onset / all alarm episodes |
| First-warning lead time | Median, across warned seizures, of the earliest successful alarm's lead time |
| Average precision | Positive-window ranking performance; compare with positive-window prevalence |
| ROC-AUC | Complementary ranking metric for comparison with previous work; not sufficient alone under imbalance |
| Window precision/recall/F1 | Performance at the validation-selected probability threshold, before alarm streak/refractory processing |
| Balanced accuracy, MCC, specificity | Supporting window diagnostics; no ordinary accuracy headline |
| Confusion counts | TP, FP, FN, TN expose the actual class counts |
| Brier score | Mean squared probability error; lower is better |
| Brier skill score | `1 - model Brier / reference Brier`; positive beats the fixed training-prevalence reference |
| Log loss | Additional proper probabilistic loss; lower is better |
| Calibration table/plot | Predicted probability versus observed positive-window fraction in fixed-width bins, with bin counts |

The reference probability is the **unweighted eligible training-window prevalence**. Its Brier loss is evaluated on the same held-out observations as the model. It is never fitted to validation/test labels. This baseline is useful but does not replace future moving-average or circadian forecasters. Class weighting does not produce calibrated probabilities automatically. Calibration is evaluated here, not fitted.

Undefined metrics are JSON `null` / empty CSV cells, not fabricated zeros. For example, AP/ROC-AUC are undefined in a single-class patient, precision is undefined when no positive decisions occur, and sensitivity is undefined when no eligible events exist. Patient macro means give each patient equal weight and state how many patients had a defined metric; pooled metrics weight all eligible observations/events.

## Time and event conventions

For decision time `t`, horizon `H` and occurrence period `P`, a successful warning targets onset in `[t+H, t+H+P)`. Bounds are half-open. An event is counted once even when multiple windows/alarms match it. Multiple target events may be matched by one alarm.

Streaks reset across missing decision gaps. The refractory timer remains in effect across those gaps within a run; all alarm state resets at a run boundary. The warning state lasts from alarm issuance until `t+H+P`, including the warning horizon. This definition measures the burden of being warned and is distinct from counting only the occurrence interval.

Eligible monitoring support is explicitly the union of forward cells `[t, t+min(stride, window))` for available eligible decisions. Overlapping waveform windows do not multiply exposure; excluded gaps do not contribute exposure. Time in warning intersects those cells with warning intervals and accounts for fractional cells. These cells are an evaluation convention, **not proof of continuous clinical monitoring**. The older `fraction_eligible_decisions_in_warning` field remains for compatibility; the new duration-based metric is `time_in_warning_fraction_eligible`.

An event is eligible if at least one retained decision can target it. The event ledger also lists target events in evaluated recordings with no eligible decision. Recordings with no retained predictions are outside this report; the preparation manifests remain the inventory of all excluded recordings/windows. Eligibility does not establish that every event had a complete preictal interval or enough consecutive decisions to trigger an alarm. Event counts and coverage must accompany sensitivity.

## Splitting, tuning and uncertainty

- Patients remain disjoint across training, validation and test.
- Grouped inner CV selects model/features on training patients only.
- Validation chooses the alarm threshold under the configured false-alarm budget.
- The selected model and threshold are frozen for final test evaluation.
- Reports use patient-cluster percentile bootstrap intervals, resampling complete patients rather than treating adjacent windows as independent.
- The default is 1,000 replicates, 95% intervals, and a minimum of five evaluated patients before intervals are emitted. Five is a pragmatic suppression rule, not a power calculation or guarantee of reliability.
- Bootstrap intervals describe held-out patient variation conditional on the fixed trained model. They do not include model-refitting uncertainty. Undefined bootstrap replicates are excluded per metric and valid replicate counts are reported.
- Validation-set scores reflect threshold selection and are not the final unbiased performance claim. Looking at test curves must not lead to retuning on that same test set.

No surrogate significance test, full-day clinical FAR, external validation or patient-specific fine-tuning has been implemented. More real patients, continuous coverage and a predeclared study protocol are necessary for those claims.

## Configuration and modules

`validation` in `config/config.yaml` controls calibration bins, bootstrap replicate count, confidence level, minimum patients for intervals, bootstrap seed and plot output. Forecast horizon, target seizure types, window stride and alarm rules are inherited from their existing configuration sections, so metrics use the same definitions as training.

- `src/validation/metrics.py`: validated window inputs, classification/probability scores, calibration bins.
- `src/models/evaluation.py`: alarm episodes, event matching, exposure and validation threshold selection.
- `src/validation/report.py`: patient reports, cluster uncertainty, curves and files.
- `src/training/runner.py`: automatically writes reports for validation and test after fitting.
- `python -m scripts.validate_model`: regenerates reports from a completed run's saved predictions, fixed threshold and saved configuration; no fitting or threshold selection.

Changing labels or alarm definitions is a new experiment. The standalone report command intentionally reads the saved run configuration rather than applying today's configuration to old predictions.

## Outputs and use

After training, look under `outputs/training/<run>/validation_reports/validation` and `test`:

- `REPORT.md`: readable summary.
- `metrics.json` and `metrics.csv`: aggregate metrics, definitions and uncertainty.
- `per_patient.csv`: individual results.
- `events.csv` and `alarms.csv`: auditable event/warning records.
- `calibration.csv`, `precision_recall.csv`, `roc.csv` and `curves.png`: numerical curves and figure. Ranking curves are omitted for single-class input.
- `complete.json`: successful report completion marker.

```powershell
.venv/Scripts/python.exe -m scripts.validate_model --run-dir "outputs/training/<completed-run>"
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m tests.smoke.training
```

Standalone regeneration creates a new timestamped directory, preserving earlier reports. Older training runs without a stored reference use the saved training-window metadata to recover training prevalence. Only use trusted local training artifacts.

Engineering tests cover hand-calculated metrics, reference forecasts, single-class and no-alarm cases, probability validation, calibration endpoints, forecast boundaries, unique event counts, exposure/gaps, fractional warning duration, reproducible patient bootstrap and end-to-end synthetic fitting. These are not real-patient efficacy results.

## Verification completed 2026-09-16

- 32 automated tests passed.
- Full synthetic EDF -> preprocessing -> features -> three-fold patient-grouped fitting -> validation/test reporting passed on five synthetic subjects (235 eligible windows).
- The resulting test report has 47 windows and two eligible events. Synthetic patterns were deliberately easy to learn; its high scores must not be used as evidence of real seizure prediction.
- Confidence intervals were correctly withheld for the single-patient test partition. A separate known-answer test exercises reproducible bootstrap intervals across five patient clusters.
- Standalone report regeneration from the completed run succeeded without fitting.
- Actual-fit compatibility checks for all three model types on 240 patient-125 rows still passed. No real population efficacy evaluation has occurred.

Artifacts: `outputs/smoke_tests/20260916T110305138199Z/SMOKE_RESULTS.json`; detailed reports under its `synthetic_training/20260916T110321827006Z_a5445ea1580981d6/validation_reports/` directory.
