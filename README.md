# EEG seizure prediction

Open **[EEG_Seizure_Prediction.ipynb](EEG_Seizure_Prediction.ipynb)**. All processing, model and evaluation code is visible inside it. The model is a **deep CNN + stacked LSTM receiving EEG waveforms**, with no handcrafted feature table or shallow classifier.

## Folder layout

```text
seizure_prediction/
├── README.md
├── EEG_Seizure_Prediction.ipynb
├── source/                         Original participant data
│   └── sub-001/ses-01/eeg/...
└── intermediate/
    └── 20261017_120000_all_seizures_abc123/  One independent session
        ├── processed/              Filtered waveform arrays and window metadata
        ├── model/                  model.pt: weights, settings, normalization, threshold
        └── report/                 report.md, metrics, predictions and learning history
```

`.venv` and `.git` are Python/Git tooling folders. No separate source-code, documentation or auditing directory is required. The `source` folder means **original data**, not Python implementation code.

## Open Jupyter

In PowerShell, change into this project folder. Reuse the existing environment. For a fresh setup, run:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install "mne>=1.5" "numpy>=1.24" "pandas>=2" "scipy>=1.10" "scikit-learn>=1.3" "torch>=2" "matplotlib>=3.7" "notebook>=7" "ipykernel>=6"
.venv/Scripts/python.exe -m ipykernel install --user --name seizure-prediction --display-name "Python (seizure_prediction)"
```

Start the application:

```powershell
.venv/Scripts/python.exe -m notebook .
```

Open **EEG_Seizure_Prediction.ipynb** and choose **Python (seizure_prediction)**. Anaconda/JupyterLab and VS Code work with an environment containing the same packages. Starting Jupyter does not execute the cells. Scikit-learn is used for numerical metrics only; model fitting uses PyTorch.

## Original EEG files: where to put them

Use [SeizeIT2, OpenNeuro ds005873](https://openneuro.org/datasets/ds005873/versions/1.1.0), also listed as [NEMAR on005873](https://nemar.org/dataset/on005873). Download actual EDF files, EEG JSON sidecars and matching events TSVs. Retain dataset licence/provenance metadata. Git/git-annex pointers and streaming-only caches are not EDF replacements.

Put participant directories **directly inside `source/`**:

```text
source/sub-001/ses-01/eeg/
    sub-001_ses-01_task-szMonitoring_run-01_eeg.edf
    sub-001_ses-01_task-szMonitoring_run-01_eeg.json
    sub-001_ses-01_task-szMonitoring_run-01_events.tsv
```

Keep original names and run/session folders. Avoid an extra wrapper such as `source/ds005873/sub-001/`. The existing local participant files have been moved from `data/raw/` into `source/`; waveform contents are unchanged. Existing non-EEG originals, if present, are retained but never read by this notebook.

To download EEG only for the first ten people, install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), then run from the project folder:

```powershell
New-Item -ItemType Directory -Path source -Force | Out-Null
foreach ($number in 1..10) {
    $subject = "sub-{0:D3}" -f $number
    aws s3 sync --no-sign-request "s3://openneuro.org/ds005873/$subject/" "source/$subject/" --exclude "*" --include "*/eeg/*"
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $subject; fix the connection and rerun." }
}
```

No AWS credentials are needed for this public download. The include rule retains all files inside EEG folders, including events and JSON. Change the participant range for more people; `SUBJECTS=None` discovers them automatically. The public bucket is not pinned to a release: keep the downloaded metadata and record the date/version used. See the [AWS sync reference](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html) or [NEMAR download guide](https://docs.nemar.org/cli/guides/downloading/) for alternatives.

## Run manually and choose channels

1. Run the import cell with **Shift+Enter**.
2. Edit **section 2**. Start with `RUN_TRAINING=False` and a few `SUBJECTS` or exact `RECORDINGS` if you want to inspect preparation first.
3. Choose `EEG_CHANNELS`. The inventory prints available names and compatible participants before creating processed files.
4. Run **Start a session** once, then continue in order. Inspect the waveforms, exclusion counts and patient/class counts.
5. For training, use `RUN_TRAINING=True` and enough eligible people, review `PATIENT_SPLITS`, restart the kernel and run from the top. Wait for **SESSION COMPLETE**, then open the printed report path.

The default uses **two EEG inputs for every example**, in this order:

```python
EEG_CHANNEL_MODE = "local_cross"
EEG_CHANNELS = None
SUBJECTS = None
```

1. `local_ear`: the recorded `BTEleft SD` or `BTEright SD` for that patient.
2. `cross_head`: the recorded `CROSStop SD` from the same EEG file.

The current ten patients provide 105 compatible recordings: 68 left-setup and 37 right-setup. The inventory displays each actual source-channel pair and its left/right side. The model receives filtered waveforms with shape **2 channels × 3,840 samples** per default 30-second example. No channel is synthesized and no polarity is changed. A recording without cross-head, or with an ambiguous local side, stops inventory rather than being silently skipped. Bilateral-only setups in the full dataset need a separately configured experiment.

For a cross-head-only comparison, set `EEG_CHANNEL_MODE="exact"`, `EEG_CHANNELS=["CROSStop SD"]`, and `SUBJECTS=None`. In exact mode, `EEG_CHANNELS=None` uses shared exact channel names. Other exact combinations and compatible patient lists are explained in notebook section 2. Exact mode never substitutes left/right channels.

These are voltage differences between behind-ear electrodes; `CROSStop SD` is not a vertex/Cz electrode. See the [dataset placement description and Figure 2](https://www.nature.com/articles/s41597-025-05580-x). Pooled `local_ear` means the clinically selected side, not a fixed left/right device. The actual source names, laterality and ordered mapping are saved per recording in `processed/recordings.json`, `report/session.json`, and `model/model.pt`. Whole-patient stratification also considers recorded side, with split membership printed for inspection. New channel modes need a new session and retraining; previous sessions remain unchanged.

The loader opens only `sub-*/ses-*/eeg/*_eeg.edf`, associated EEG JSON and event TSV files. ECG/EMG/movement originals stay on disk and are never opened by this workflow. Every channel selection still trains the waveform CNN + LSTM; there is no shallow-learning fallback.

## What each session stores

Each execution of **Start a session** creates a unique timestamp/name/suffix. Sessions are independent, including their processed waveforms; they do not reuse another session's cache. Rerun from the beginning after parameter changes. Old sessions remain until you delete them.

- **processed:** one `.npy` per usable EEG recording, `windows.csv` with labels/exclusions, and `recordings.json` with file/channel metadata. Arrays contain retained **examples × channels × samples**, in float32 microvolts.
- **model:** `model.pt`, including best weights, architecture, channel order, preprocessing, training normalization and selected threshold. The notebook includes a reload example. A partial fit may have no finalized threshold.
- **report:** readable `report.md`, machine-readable metrics/predictions, epoch history/plot, and `session.json` with settings, progress or an error traceback. `stage: complete` confirms saving and matching reloaded validation predictions.

Signal processing and training use bounded windows/mini-batches from disk. RAM holds the current batches, metadata and model. Models and longer windows still need working memory. A new session requires disk space for another set of waveform arrays. Delete a whole unwanted session after closing its kernel/files.

## Prediction method

Defaults use a 30-second observed input, 0.5–40 Hz causal band-pass, 128 Hz output and a 30-second decision stride. An optional notch/amplitude rejection threshold is exposed. Anti-alias downsampling and filtering deliberately remove configured frequency content; they are not lossless raw-data storage or complete artifact removal. Original EDFs stay unchanged.

The network learns waveform representations with three convolutional blocks, then a stacked LSTM reads their order within the observed window. There is no fixed mean/power/entropy feature summary. Increasing `WINDOW_SEC` gives more past context at a computation/memory cost; hidden state is not carried across windows or recordings.

The target remains seizure onset in `[t + 30 min, t + 40 min)` by default. Choose which seizure codes make a positive target in section 2:

```python
SEIZURE_GROUP = "focal"  # All focal subtypes; default "all" includes any seizure.
STRATIFY_PATIENTS = True
```

Shipped defaults are `SEIZURE_GROUP="all"`, `STRATIFY_PATIENTS=True`, `PATIENT_SPLITS=None`, and `SESSION_NAME="all_seizures"`. Notebook section 2 contains seven ready-to-copy target/session configurations. Presets include `"all"`, `"focal"`, `"focal_aware"`, `"focal_impaired_awareness"`, `"generalized"` and `"custom"`. To select exactly one annotation code:

```python
SEIZURE_GROUP = "custom"
SEIZURE_PREFIXES = ["sz_foc_ia_nm"]
SEIZURE_MATCH = "exact"  # Use "prefix" to include that code's descendants too.
```

The inventory prints observed codes/counts/patients and which codes are selected. The current 33 seizure annotations all have focal codes, so `"all"` and `"focal"` currently produce the same targets. See the notebook's selector guide and the [dataset code definitions](https://raw.githubusercontent.com/OpenNeuroDatasets/ds005873/refs/heads/main/events.json).

This is binary forecasting: 1 means a selected future seizure, 0 means eligible background, and -1 means excluded. Nonselected seizures in the forecast target interval are excluded rather than treated as background. Unknown future coverage, all seizure/postictal inputs, warning-gap seizures and impedance periods remain excluded. Selected event IDs/types are target metadata in `processed/windows.csv`, not model predictors. Window labels and alarm metrics use the same selection rule. Change the target in a new session and retrain; old sessions keep their original labels and model settings.

All windows from a patient stay in one train/validation/test partition. Training-only channel normalization and class weights avoid learning those quantities from held-out people. Validation selects early stopping and alarm threshold; test is evaluated after those choices are fixed.

At least three distinct eligible people are structurally necessary, with both classes and sufficient selected eligible seizures in every partition. Two people work for preparation only. The automatic split uses a seeded search around 60/20/20 to balance selected subtype event counts and positive/background windows across whole patients. Small cohorts are searched exhaustively at those sizes; larger cohorts use a bounded search. Exact balance is not guaranteed. No model scores enter the split selection, and there is no random-window fallback. `STRATIFY_PATIENTS=False` restores sorted draft splitting; explicit `PATIENT_SPLITS` always take priority.

Inspect the printed subtype counts and `report/patient_splits.json`. A subtype occurring in fewer than three eligible people cannot appear in every partition. Narrower targets may leave too little data for training; the notebook explains the failure instead of silently relaxing the study. Many positive windows pointing to the same original event count once, including windows that target several selected seizures.

Raw-waveform deep learning does **not** guarantee better scores. Inspect held-out average precision against prevalence, event sensitivity and false alarms together. A quiet threshold can meet a false-alarm budget while missing every seizure. Evaluation covers eligible windows only, not continuous clinical monitoring; no patient-baseline adaptation is implemented.

The notebook has debugging checks and reports errors inside the session. Verification passed on four synthetic participants, two-patient preparation, and four real EEG recordings (1,479 eligible windows, 40 positive windows). The real check used two training epochs and a 120-second stride; saved-model predictions matched 532 held-out rows in a fresh Python process. Channel order, training-only normalization, noise filtering and protection against overwriting sessions were also checked. Temporary testing files were removed. These checks verify software behavior, not improved prediction performance.

The seizure selector also passed known-answer tests for exact/prefix matching, unrelated-seizure exclusions, unique event counting and patient stratification. A full focal-aware run on five EEG recordings from four people completed with 1,231 eligible windows and 35 positives, including model reload and matching split/evaluation event counts. Rare-target failures and the larger-cohort split search were checked separately.

The default two-channel mapping was checked against all 105 local EEG headers (68 left-setup, 37 right-setup). A two-epoch check on four real recordings from four patients completed processing, CNN–LSTM training, evaluation and model reload with both local-ear sides represented. Persisted waveform values matched the selected EDF channels in order; a fresh process reproduced 413 validation predictions. Missing/ambiguous channel rejection, exact-mode comparisons, side-aware patient splitting and both-channel plots were checked. Temporary test files were removed. This verifies execution, not forecasting accuracy.
