# Seizure forecasting research pipeline

Configurable EEG, ECG, EMG and motion preprocessing, feature extraction, patient-separated training, forecasting validation and a local experiment dashboard.

## What this repository contains

This repository contains **source code, configuration templates, tests and documentation only**. It does not contain participant recordings, annotations, prepared features, trained models, experiment reports or other generated data.

The workflow is a research tool, not a clinical device. Its evaluation is conditional on eligible observation windows and does not establish clinical safety or effectiveness.

## Before you start

### 1. Create a Python environment

The commands below use PowerShell on Windows and Python 3.10 or later:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Download SeizeIT2 yourself

The input data for this project is the **SeizeIT2** wearable epilepsy dataset. Download it from the official [NEMAR SeizeIT2 dataset page](https://www.nemar.org/dataset/on005873) (DOI: `10.82901/nemar.on005873`) or its linked official distribution methods. The full dataset is large, so obtain only the participants and recordings needed for your experiment where possible.

After downloading, place the BIDS participant folders and their accompanying recording/event files under `data/raw/`, for example:

```text
data/raw/
└── sub-001/
    └── ses-01/
        └── ... recording and event files ...
```

Do **not** commit anything in `data/`. It is intentionally excluded from Git because it can contain sensitive participant data and large derived artifacts. You are responsible for meeting the dataset licence, access conditions and institutional data-governance requirements.

### 3. Start the local dashboard

From the project directory:

```powershell
.venv/Scripts/python.exe dashboard.py
```

Open **http://127.0.0.1:8765**. Select **Load synthetic test → Start experiment** for a tiny synthetic end-to-end test. Use **Reset defaults** before configuring real patient data. [Dashboard guide](docs/DASHBOARD.md)

For a fast check on the downloaded real recordings, choose **Load real-data check → Start experiment**. It uses four exact recordings, explicit patient partitions and a separate 5–10-minute forecast target. [Why the earlier full run failed and how this check works](docs/REAL_DATA_CHECK.md). The synthetic button is now named **Load synthetic test**.

## Typical workflow

1. Download and locally store an approved SeizeIT2 subset in `data/raw/`.
2. Configure modalities, forecast horizon and patient partitions in `config/config.yaml` or the dashboard.
3. Run preprocessing and feature preparation.
4. Use the readiness check to verify disjoint patient partitions, class coverage and grouped folds.
5. Start training only after readiness succeeds; model selection happens within training-subject cross-validation, then held-out validation selects the alarm threshold, followed by one test evaluation.
6. Treat saved metrics as research results and inspect their documented limitations before drawing conclusions.

## Directory map

```text
seizure_prediction/
├── dashboard.py              # Convenient dashboard launcher
├── README.md                 # Project overview and commands
├── requirements.txt          # Python dependencies
├── config/                   # Default configuration and legacy reference
├── scripts/                  # Production command-line entry points
│   ├── preprocess.py
│   ├── prepare_features.py
│   ├── train.py
│   └── validate_model.py
├── src/                      # Reusable implementation
│   ├── data/                 # Recording discovery and prepared-data loading
│   ├── preprocessing/        # Alignment, filtering, windows and labels
│   ├── features/             # Modality-specific feature extraction
│   ├── training/             # Patient splits, model registry and fitting
│   ├── models/               # Feature selection and alarm evaluation
│   ├── validation/           # Metrics, uncertainty and reports
│   ├── dashboard/            # API, jobs and browser assets
│   ├── pipeline/             # Preprocessing orchestration
│   └── utils/                # Configuration loading
├── tests/                    # All test/verification code and test notes
│   ├── test_*.py             # Automated regression tests
│   ├── smoke/                # Small end-to-end tests
│   ├── checks/               # Optional prepared-data audits
│   └── docs/                 # Historical test status notes
├── docs/                     # Current usage and methodology guides
├── data/
│   ├── raw/                  # Original recordings; preserved
│   └── interim/              # Prepared arrays, features and manifests
└── outputs/                  # Generated models, logs, figures and reports
    ├── dashboard/            # Isolated dashboard experiments
    ├── training/             # CLI training experiments
    ├── smoke_tests/          # Synthetic fixtures and test results
    ├── reports/              # Preprocessing/feature audit summaries
    ├── figures/              # Standalone figures/PDFs, if produced
    └── logs/                 # Standalone operational logs
```

Environment/editor folders such as `.venv`, `.vs` and `.ipynb_checkpoints` are local tooling state, not pipeline source. `.gitignore` excludes them, patient recordings and generated artifacts.

## Commands

Use **module commands from the project root** for consistent imports and configuration paths.

| Task | Command after `.venv/Scripts/python.exe` |
|---|---|
| Preprocess | `-m scripts.preprocess --config config/config.yaml` |
| Extract features | `-m scripts.prepare_features --config config/config.yaml` |
| Check readiness without fitting | `-m scripts.train --config config/config.yaml` |
| Train and evaluate | `-m scripts.train --config config/config.yaml --start-training` |
| Regenerate prediction reports | `-m scripts.validate_model --run-dir outputs/training/<run>` |
| Regression tests | `-m unittest discover -s tests -v` |
| Small training smoke test | `-m tests.smoke.training` |
| Test live dashboard | `-m tests.smoke.dashboard --eeg-only` |
| Audit waveforms | `-m tests.checks.preprocessing --config config/config.yaml` |
| Audit features | `-m tests.checks.features --config config/config.yaml` |

Former root scripts have moved into `scripts/` and `tests/`. The top-level dashboard command remains unchanged. [Testing guide](tests/README.md)

## What goes where?

- Edit parameters in `config/config.yaml` or the dashboard.
- Keep algorithms in `src/` and CLI argument handling in `scripts/`.
- Put regression tests, smoke tests and test documentation in `tests/`.
- Keep original EDF/JSON/TSV recordings in `data/raw/`.
- Put generated reports in `outputs/`. Experiment figures and reports stay beside their saved configuration and model.
- Keep manifests and prepared arrays/features together in `data/interim/` or each dashboard experiment's `interim/`.

Existing experiment directories are preserved because manifests contain absolute artifact paths. Audit summaries formerly in `data/interim/` now live in `outputs/reports/preprocessing/` and `outputs/reports/features/`. [Output directory guide](outputs/README.md)

## Qualification and clean starts

`qualification` in the configuration is enabled by default. It excludes recordings with missing required files or preprocessing errors, recordings with no eligible forecast windows, and patients that are selected but not assigned to train, validation, or test. Each exclusion is printed as `FILTERED ...` in the pipeline log and saved in the experiment's `interim/qualification.json`.

Use **Clear generated data & logs** in the dashboard for a fresh run. After its confirmation, it clears generated interim artifacts, shared prepared snapshots, reports, training outputs, smoke-test artifacts, and dashboard experiment history. It preserves `data/raw/`, source code, documentation, the configuration files, and `outputs/README.md`.

## Documentation

- [Preprocessing, timing and labels](docs/PREPROCESSING.md)
- [Training architecture](docs/TRAINING_PIPELINE.md)
- [Validation metrics and research sources](docs/VALIDATION_PIPELINE.md)
- [Feature and model research decisions](docs/RESEARCH_DECISIONS.md)
- [Dashboard and API](docs/DASHBOARD.md)
- [Testing and verification](tests/README.md)

## Scientific status

The engineering workflow has passed automated and small synthetic checks. Full real-population training remains pending additional patients and explicit patient partitions.

The default forecast targets onset 30–40 minutes after the decision time. Publisher-relative synchronization remains an assumption, and alarm rates apply to eligible observation windows. Synthetic scores do not demonstrate clinical forecasting performance.

