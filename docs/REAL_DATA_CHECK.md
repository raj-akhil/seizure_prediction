# Real-data pipeline check and diagnosis (2026-09-16)

## Why the attempted full run stopped

There are five patient directories total, `sub-121` through `sub-125`. Downloading/copying patients does not populate `training.splits`; the default train, validation and test lists remain intentionally empty until an experiment assigns them.

The latest saved failing dashboard run (`6c0f47bc07e14a98976fc6a6763e2ed7`) stopped during preprocessing with `Missing ecg`, before training. A filename/annotation inventory found:

| Patient | Discovered runs | Recordings with a missing selected modality or event file | Recorded seizure onsets |
|---|---:|---:|---|
| sub-121 | 111 | 111 | No seizures in the four currently present event files; cannot infer absence in missing annotations |
| sub-122 | 70 | 0 | Four annotated seizures |
| sub-123 | 85 | 1 (motion absent in run-12) | Two annotated seizures |
| sub-124 | 84 | 0 | One annotated seizure |
| sub-125 | 85 | 0 | Two annotated seizures |

This is a file inventory, not verification of every EDF's integrity. Patient 121's copy is incomplete for the selected four-modality experiment and lacks most event files.

There is also a forecast-label constraint. Both sub-123 seizure onsets (158 and 988 seconds into their respective runs) and sub-124's onset (951 seconds) occur before 30 minutes of within-run history exists. The current pipeline does not assume continuity across runs. They therefore cannot supply positive 30-minute-horizon examples. Merely filling split lists cannot resolve this.

Default minimum-event requirements matter too: the normal configuration requires at least two eligible events per outer partition. A patient with one eligible seizure is not enough for a validation/test partition under that setting.

## Small verification configuration

`config/real_data_check.yaml` points directly at four existing real recordings; no EDFs are copied and no dummy data is generated.

| Partition | Patient | Recording |
|---|---|---|
| Training | sub-122 | run-10 |
| Training | sub-125 | run-21 |
| Validation | sub-123 | run-81 |
| Test | sub-124 | run-47 |

All four chosen recordings have EEG, ECG, EMG, motion and event files. Selection uses `dataset.recordings`, a list of full recording IDs, so selecting different runs from different patients does not inadvertently select their cross-product.

The engineering check keeps all four modalities, the existing filters, 256 Hz output and five-second waveform windows. It uses **60-second stride** to process fewer windows, rather than lowering signal sampling rates. ECG cardiac features still use preceding 60-second histories. The target is **5–10 minutes ahead**, explicitly separate from the default 30–40-minute target.

It uses two patient-grouped inner folds, one minimum positive event per outer partition, small grids for all three classifiers, ten selected features, and one high-score window to trigger an alarm. Threshold candidates include 1.0 so an unhelpful model can select no warnings. The default false-alarm budget is retained. Bootstrap intervals are disabled for this tiny cohort. These settings are for execution verification, not a scientific performance study.

The original `config/config.yaml` retains its 30-minute warning horizon, three folds and two-event partition requirement. No data split or event-count requirement was silently relaxed for that experiment.

## Result

- Complete real-data workflow passed in **15.41 seconds** on this machine.
- Four recordings, 141 eligible feature rows, 19 positive rows.
- Preprocessing, features, readiness, grouped fitting, validation threshold selection and held-out test reports completed.
- The selected random forest used threshold 1.0 and issued no test alarms (event sensitivity 0). This run verifies execution; it does not demonstrate a useful forecasting model.
- All 41 automated tests passed, including exact recording selection and missing-file detection before waveform processing.
- The sampled, seizure-enriched recordings and one validation/test seizure each cannot support clinical performance estimates. Sparse-window exposure also cannot establish continuous false-alarm rates.

Saved execution summary: `outputs/reports/real_data_check/latest.json`.
Saved experiment: `outputs/dashboard/ceedc22e5a5048ecb5a2436d980303f3/`.

## Run it yourself

Start the dashboard, refresh its page, click **Load real-data check**, then **Start experiment**. The editor shows the exact selected recordings and shortened forecast settings. This is distinct from **Load synthetic test**.

For a bounded CLI check with the dashboard running:

```powershell
.venv/Scripts/python.exe -m tests.smoke.real_data --timeout-sec 180
```

The check requests cancellation if it exceeds three minutes. It does not launch the entire downloaded dataset.

Before a full scientific experiment, finish incomplete downloads, inventory usable pre-seizure coverage for the intended horizon, then assign disjoint patient partitions with sufficient eligible events. Evaluate complete representative recordings rather than interpreting this selected pilot as final performance.
