# Seizure prediction — two-notebook workflow

Start with the two notebooks in this folder. They contain editable parameters, explanations, expected outputs, plots and debugging checks. You do not need the older dashboard or command-line tools.

1. **[01_prepare_data.ipynb](01_prepare_data.ipynb)** discovers participant recordings, checks files, filters/windows signals, assigns forecast labels and extracts trainable features.
2. **[02_train_model.ipynb](02_train_model.ipynb)** reads that dataset, checks patient splits, fits/compares models, chooses an alarm threshold, evaluates held-out patients and saves the trained pipeline.

Run notebook 1 from top to bottom, then notebook 2. Use the same `DATASET_NAME`. Change parameters near the top; after a change, restart the kernel and run from the beginning.

**The GitHub repository contains code, notebooks and documentation only. It does not include the recordings, a Python environment, prepared features or trained models. Download data separately using step 2 below.**

## 1. Set up Python

Install [Python 3.11](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads). Clone/download this project, open PowerShell, and change into the folder containing this README. Run:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m ipykernel install --user --name seizure-prediction --display-name "Python (seizure_prediction)"
```

If `.venv` already exists on your machine, reuse it and run the installation command to update its dependencies. There is no need to activate it: the commands use its Python executable explicitly. The optional LSTM dependency, PyTorch, is included in `requirements.txt`; initial installation may take time and disk space.

Run every command below from the **project root**, not from `src/` or `data/`.

## 2. Download SeizeIT2 and put it in `data/raw/`

### Dataset and download tools

Use the official **SeizeIT2** dataset: [OpenNeuro `ds005873`](https://openneuro.org/datasets/ds005873/versions/1.1.0), also available as [NEMAR `on005873`](https://nemar.org/dataset/on005873). Review its licence/access conditions and cite the dataset when publishing results. Download the original EDF recordings and their annotations/metadata; this project does not accept screenshots or a Zarr-only streaming cache as replacements.

The PowerShell instructions below download directly from OpenNeuro's public S3 copy into the correct project folders. Install [AWS CLI v2 for Windows](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), reopen PowerShell if necessary, and check:

```powershell
aws --version
```

`--no-sign-request` uses public access; you do not need to configure AWS credentials. The [`aws s3 sync` documentation](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html) explains the command. The bucket is the current public copy, rather than a pinned release: record your download date and retain the downloaded metadata for reproducibility. For a specific release or alternative download method, use the dataset pages above or the [official NEMAR download guide](https://docs.nemar.org/cli/guides/downloading/).

### Download the first ten participants

From the project root, run this PowerShell block. It downloads every sensor, recording and accompanying file for `sub-001` through `sub-010`, preserving filenames and directories:

```powershell
New-Item -ItemType Directory -Path data/raw -Force | Out-Null

foreach ($number in 1..10) {
    $subject = "sub-{0:D3}" -f $number
    aws s3 sync --no-sign-request "s3://openneuro.org/ds005873/$subject/" "data/raw/$subject/"
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $subject. Rerun after fixing the connection." }
}
```

This is a substantial download. The ten-participant copy inspected during development was about **9.88 GiB**, before generated data; allow additional space for prepared arrays and models. That size is a local example, not a guarantee for another dataset version. Rerun the same command after interruption: `sync` skips files it considers unchanged and retries missing/changed files. The preparation notebook checks the downloaded recording headers and metadata before processing.

To add a specific participant later:

```powershell
aws s3 sync --no-sign-request "s3://openneuro.org/ds005873/sub-011/" "data/raw/sub-011/"
```

Change `1..10` to the desired range to download a larger cohort. You can also choose individual participants rather than a range. Do not download only EEG if you plan to enable ECG, EMG and motion in the notebook.

### Required local folder layout

Participant folders must be **direct children of `data/raw/`**. For one recording, the expected layout is:

```text
data/raw/
└── sub-001/
    └── ses-01/
        ├── eeg/
        │   ├── sub-001_ses-01_task-szMonitoring_run-01_eeg.edf
        │   ├── sub-001_ses-01_task-szMonitoring_run-01_eeg.json
        │   └── sub-001_ses-01_task-szMonitoring_run-01_events.tsv
        ├── ecg/
        │   ├── sub-001_ses-01_task-szMonitoring_run-01_ecg.edf
        │   └── sub-001_ses-01_task-szMonitoring_run-01_ecg.json
        ├── emg/
        │   ├── sub-001_ses-01_task-szMonitoring_run-01_emg.edf
        │   └── sub-001_ses-01_task-szMonitoring_run-01_emg.json
        └── mov/
            ├── sub-001_ses-01_task-szMonitoring_run-01_mov.edf
            └── sub-001_ses-01_task-szMonitoring_run-01_mov.json
```

Keep each `.edf` signal file, its `.json` sidecar and the matching `_events.tsv` annotation file. Do not rename files, flatten sensor folders, mix runs, or copy Git/git-annex pointer files instead of real recording contents. The event timeline uses the EEG recording origin even if you select other signal types.

If you download through a browser or another tool, extract/copy the `sub-*` directories into `data/raw/`, with the structure above. If the download contains dataset-level files such as `dataset_description.json`, `participants.tsv`, `participants.json` and `events.json`, retain those alongside the participant folders for provenance. Avoid an extra wrapper such as `data/raw/ds005873/sub-001/`; the notebook's default `RAW_DIR` is `data/raw/`. Do not copy the downloaded dataset's `.git` directory into this project.

Check placement from PowerShell:

```powershell
Get-ChildItem data/raw -Directory
Get-ChildItem data/raw -Recurse -File -Filter *.edf | Measure-Object
Get-ChildItem data/raw -Recurse -File -Filter *_events.tsv | Measure-Object
```

The development copy had **10 participants, 420 EDF files and 105 event TSV files**. Other selections/releases may differ. Notebook 1's inventory gives the selected recording count, annotations and missing-file problems; file counts alone do not prove integrity.

## 3. Run preprocessing, then training

Start Jupyter from the project root:

```powershell
.venv/Scripts/python.exe -m notebook .
```

Open `01_prepare_data.ipynb` and choose **Python (seizure_prediction)** as the kernel. In VS Code, opening the notebook and selecting `.venv/Scripts/python.exe` works too.

1. In notebook 1, edit the parameters cell. Keep `RAW_DIR = ROOT / "data" / "raw"`. `SUBJECTS = None` discovers all available participants; a list such as `["sub-001", "sub-002"]` limits preparation. `RECORDINGS` can select exact full recording IDs from the inventory.
2. Run every cell in order. Inspect the source-file inventory, waveform preview, final class counts and exclusions. Wait for **PREPARATION COMPLETE** and a saved `handoff.json`.
3. Open `02_train_model.ipynb` with the same kernel environment and the same `DATASET_NAME`. Prepared files persist on disk; the two notebooks do not need to share Python variables or remain open simultaneously.
4. Review the patient assignments and readiness tables. Set explicit `PATIENT_SPLITS` when defining a study, or review the default sorted 70/15/15 draft. Change model grids/feature selection near the top, then run all cells in order.
5. Wait for **TRAINING COMPLETE**. Inspect the validation/test reports and the saved model. `complete.json` confirms saving and reload verification succeeded.

With the default **three inner folds**, training needs at least **five eligible patients**: at least three training patients, plus distinct validation and test patients. Every partition/fold also needs the required classes/events; five folders alone are not sufficient. Two patients can check preparation and individual fitting functions, but the production training notebook correctly refuses a separate train/validation/test study with only two people. It never falls back to random-window splitting.

The normal notebook run fits the selected models and evaluates held-out patients; run it deliberately. After changing parameters, restart the kernel and run from the top to avoid stale variables. For a first preprocessing trial, select a few exact recordings; for full training, select enough eligible patients. Larger cohorts may need more RAM or a smaller experiment.

## Project structure

```text
seizure_prediction/
├── README.md                   Start here
├── 01_prepare_data.ipynb        Recordings → trainable features
├── 02_train_model.ipynb         Features → model and evaluation
├── requirements.txt            One dependency list, including Jupyter
├── data/
│   └── raw/                    Original participant folders: sub-001, sub-002, ...
├── documentation/              Methodology, debugging and maintenance guides
├── audit/
│   ├── runs/                   Prepared data, logs, models, reports and checks
│   ├── archive/                Older generated files and temporary build material
│   └── maintenance/            Layout migration record and raw-data inventory
└── src/                        Python implementation used by the notebooks
    ├── config/                 Advanced defaults; normal parameters are in the notebooks
    ├── commands/               Optional CLI tools and dashboard launcher
    ├── tests/                  Internal regression and end-to-end checks
    └── ...                     Signal, feature, model and evaluation code
```

`.git` and `.venv` are Git/Python tooling folders. Keep `src/`: it contains the algorithms the notebooks import. The notebooks must stay inside this project.

## Key parameters and future participants

| Parameter | Meaning |
|---|---|
| `DATASET_NAME` | Dataset/experiment group; use the same value in both notebooks |
| `SUBJECTS` / `RECORDINGS` | Which people/runs to prepare; `None` selects all available matches |
| `MODALITIES` / `CHANNELS` | Sensors and exact channel names; all four sensor types are enabled initially |
| `WINDOW_SEC` / `STRIDE_SEC` | Signal duration per example / spacing between examples |
| `MINIMUM_WARNING_MIN` / `OCCURRENCE_PERIOD_MIN` | Future interval used to assign forecast labels |
| `PATIENT_SPLITS` | Disjoint train, validation and test patient lists |
| `MODEL_GRIDS` / `FEATURE_K` | Models/parameter choices to compare and selected feature count |
| `MAX_FEATURE_MATRIX_GIB` | Dense feature-matrix memory guard; fitting requires additional memory |

Teaching defaults use five-second windows, **60-second stride**, native sensor rates and a **30–40 minute forecast**. At decision time `t`, a positive label targets onset in `[t + 30 min, t + 40 min)`. Invalid/unobserved periods are excluded. The notebook explains how to restore the original five-second stride and 256 Hz output. A sparse stride does not represent continuous monitoring.

New `sub-*` folders are discovered automatically with `SUBJECTS = None`. Rerun notebook 1 to include them: saved snapshots freeze their original recording selection. Then review the patient assignments in notebook 2. Training loads a dense feature table and is not out of core.

## Prepared data, logs and models

Notebook output lives under `audit/runs/notebooks/<DATASET_NAME>/`:

- `prepared/<settings-hash>/`: configuration, signal windows, features, summary and `preparation.log`.
- `handoff.json`: the exact prepared dataset read by notebook 2.
- `training/<timestamp>/`: model, configuration, predictions, metrics, plots and `training.log`.

Changing models reuses preparation. Changing participants, sensors, windows, labels or features requires notebook 1 again. A training directory's `complete.json` confirms successful model saving and reload verification. [Output guide](documentation/AUDIT.md)

Raw data and generated files under `data/` and `audit/` are excluded from Git, apart from `audit/README.md`. They will not appear after cloning from GitHub; the notebooks create output folders when run. `audit/archive/` and `audit/maintenance/` describe optional local history, not files required to execute a fresh checkout. Never place recordings, credentials or trained models in a source-code commit.

## Common problems

| Problem | What to check |
|---|---|
| `No EDF recordings` | Files are extracted and participant folders are directly under `data/raw/` |
| Missing sensor/event/channel | Complete the download and match the selected modalities/channels; retain each run's event TSV |
| Imports fail | Install `requirements.txt` and select the project's Python kernel |
| No handoff / stale features | Finish notebook 1 with the matching `DATASET_NAME`; rerun after changing preparation settings |
| Not enough patients/classes/events | Read the readiness counts; add eligible data or deliberately redesign the study |
| Memory/disk guard | Use fewer recordings/sensors, increase stride, or use a machine with more memory/storage |
| Slow processing | Check progress in `preparation.log`; completed caches can be reused on rerun |

Full tracebacks and step timings appear in `preparation.log` and `training.log`. The notebooks include waveform/feature previews and `%debug` guidance. Do not bypass failed readiness checks to obtain a score.

## Further reading and verification

[Documentation index](documentation/README.md) · [Preprocessing](documentation/PREPROCESSING.md) · [Training](documentation/TRAINING_PIPELINE.md) · [Validation](documentation/VALIDATION_PIPELINE.md) · [Testing](documentation/TESTING.md)

```powershell
.venv/Scripts/python.exe -m unittest discover -s src/tests -v
.venv/Scripts/python.exe -m src.tests.smoke.notebooks
.venv/Scripts/python.exe -m src.tests.smoke.notebooks --real
```

These are engineering checks. The real-data notebook check uses four selected EEG recordings, not the complete cohort. Results are conditional on eligible forecast windows and do not establish clinical forecasting performance. Alignment remains the documented publisher-relative timing assumption.
