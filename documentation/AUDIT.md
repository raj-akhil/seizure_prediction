# Prepared data, experiment logs and audit results

The notebooks write to `audit/runs/notebooks/<DATASET_NAME>/`.

| Path inside a dataset folder | Meaning |
|---|---|
| `prepared/<hash>/config.yaml` | Preparation settings and frozen recording selection |
| `prepared/<hash>/signals/` | Memory-mapped waveforms and signal-quality masks |
| `prepared/<hash>/experiments/` | Window timing, labels, annotations and signal reports |
| `prepared/<hash>/features/` | Features, dictionaries and final eligibility |
| `prepared/<hash>/feature_manifest.json` | Prepared file locations and row counts |
| `prepared/<hash>/dataset_summary.csv` | Per-recording row/class counts |
| `prepared/<hash>/preparation.log` | Progress, durations and exceptions |
| `handoff.json` | Completed snapshot pointer read by notebook 2 |
| `training/<timestamp>/readiness.json` | Patient/event/class counts and grouped-fold checks |
| `training/<timestamp>/model.joblib` | Fitted cleanup, selector, classifier, threshold and config |
| `training/<timestamp>/*_predictions.csv` | Fixed held-out scores and labels |
| `training/<timestamp>/validation_reports/` | Metrics, patient/event/alarm tables and plots |
| `training/<timestamp>/training.log` | Fit progress and exceptions |
| `training/<timestamp>/complete.json` | Successful saving and reload verification |

Model changes reuse a handoff. Changes to participants, signals, windows, labels or features require preparation again. Do not move individual snapshot files: manifests use absolute paths.

Other generated folders under `audit/runs/` preserve earlier dashboard/CLI experiments and engineering checks. `audit/archive/` preserves older PDFs, slides and temporary builders. `audit/maintenance/layout_migration.json` records the structural cleanup and preserved raw-file count.

Original recordings stay in `data/raw/`. Participant data, prepared arrays, models and generated audit results are excluded from Git.
