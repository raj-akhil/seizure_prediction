# Local experiment dashboard

## Start

From the project directory:

```powershell
.venv/Scripts/python.exe -m src.commands.dashboard
```

Open **http://127.0.0.1:8765** in your browser. Keep the server terminal running; use Ctrl+C to stop it. An alternative port can be supplied with `--port`. Run only one dashboard server for this workspace at a time. No additional packages or external services are needed.

## Workflow

For the downloaded real cohort, **Load real-data check** loads four exact recordings with explicit patient partitions and a sparse 5–10-minute forecast test. See [the diagnosis and real-data check](REAL_DATA_CHECK.md). **Load synthetic test** is the separate artificial-signal demonstration previously called Load quick test.

1. **Configure:** use the patient range fields (for example, training `1-50`, validation `51-75`, test `76-125`), patient checkboxes and Partition dropdowns, or preview and apply **Suggest split**. Ranges are inclusive and replace the draft cohort/split assignments, so groups cannot overlap accidentally. Other settings cover modalities/channels, forecast windows, features, models, search grids and validation. Other lists use JSON syntax; `null` selects all where supported. Advanced JSON permits adding/removing nested entries such as models and run offsets. Click Apply JSON after editing that view.
   For a fast demonstration, click **Load synthetic test**, then **Start experiment**. This loads the existing five-patient synthetic fixture with small model grids. It does not process your real dataset.
2. **Validate settings:** runs the existing configuration validator and model-registry checks. This checks structure and accepted options; data-specific channel names, complete recording coverage and split viability are checked by the actual stages.
3. **Save draft:** saves the editor configuration in this browser. Download draft JSON also provides a portable copy. Reset restores the current project default configuration in the editor. None of these actions overwrite `src/config/config.yaml`.
4. Name the experiment and select how far to run: preprocessing, features, readiness without fitting, or training plus evaluation.
5. **Monitor:** shows each stage as pending/running/complete/failed/cancelled, with live output and per-fold fit logs. Progress is actual stage/log output, not an invented percentage or estimated remaining time.
6. **Results:** shows held-out test or validation metrics, uncertainty where available, PR/ROC/calibration figures and report downloads. Preparation-only jobs show feature counts if feature extraction completed; they do not fabricate evaluation results.

The training mode runs preprocessing -> features -> readiness -> training/evaluation. Readiness retains the disjoint patient and minimum-event checks. Patient 125 alone cannot establish population train/validation/test performance. The dashboard shows the precise failure in its log.

Empty/overlapping patient partitions and too few configured training patients are rejected **before** launching preprocessing. The data-dependent readiness checks still run after feature preparation.

Use **Use these settings** on a previous job to start a new experiment with that job's configuration. Finished reports stay attached to their original settings. Configuration edits never mutate a running job or a saved model.

## Patient selectors and split suggestions

### Recover a readiness failure without repeating preprocessing

In **Monitor**, select the failed experiment and click **Repair split from prepared data**.
This reads its saved eligible-window labels and checks both classes, distinct seizure
episodes and the actual grouped training folds. It proposes a small change to patient
roles while preserving the warning horizon, feature settings, fold count and event
minimum. The resulting roles are loaded into Configure for review; no model is fitted.
If the cohort cannot meet the requirements, repair reports that limitation.

Click **Start experiment** with **Training readiness · no fitting** selected to verify
the repaired configuration. The new experiment reads the source's feature snapshot;
preprocessing and feature extraction are skipped. Results shows background/positive
window counts and eligible seizure episodes by partition, with a downloadable
`training/readiness.json` report. The original experiment and its configuration remain
unchanged.

**Reuse prepared features** loads the same snapshot without changing patient roles.
You can then select **Training + validation + test** to fit using those features.
Reused jobs have separate logs, configuration and training outputs. Retain the source
experiment's files: the new job references its prepared data instead of copying it.
Its exact patient/recording selection is frozen, so newly downloaded runs do not
silently join an old feature snapshot. Changes to modalities, channels, alignment,
windows, labels, features or cohort require **Switch to fresh preprocessing**.
Clear exact recording filters explicitly when including new downloads.

Automatic repair is allowed only before model fitting in the source job and its
prepared-data ancestors. Keep evaluated test patients fixed when comparing models.
Passing readiness establishes split feasibility, not scientific effectiveness or
representative validation coverage. Each patient's usable data may contain only one
class; combining patients can make the partition pass while still producing a small,
unrepresentative evaluation set.

The **Patients & split assignments** panel scans the configured **Raw data folder**.
After copying more data, click **Refresh patients**. The fast refresh reads BIDS
filenames and shows the cohort and missing modalities without opening sidecars,
annotation tables or EDF waveform samples. Click **Check timing & event details**
only when you need the potential-event estimates or a suggested split; that is
the slower metadata pass. Enter patient-number ranges such as `1-50` (or
comma-separated ranges such as `1-30, 41-50`) to assign Training, Validation and
Test in one step. Ranges are inclusive and must not overlap; listed IDs such as
`sub-001` are matched by their final number. You can still use each patient's
**Partition** dropdown to adjust an individual assignment. Choosing a role
removes that patient from the other roles. Deselecting a patient removes their
assignments. These controls update `dataset.subjects` and `training.splits` in
the draft, exported JSON and the next experiment's saved YAML.

**Include all downloaded patients** sets `dataset.subjects: null`; new patients
are included after refresh but remain unassigned. Turning it off freezes the
currently listed patients into an explicit list. Exact recording and run filters
still apply, including those loaded by **Load real-data check**. The table shows
matching versus total recordings and missing files for the chosen modalities.

**Suggest split** prepares a reviewable draft using the training seed and roughly
70% training / 15% validation / 15% test, adjusted for small cohorts and the
configured number of training folds. Click **Apply suggested split** to write it
into the editor. It does not change your patient selection, forecast horizon,
minimum event count or folds. Keep the test group fixed when comparing models;
suggestions do not optimize model scores or automatically stratify seizure types.

The suggestion requires complete selected recordings, enough patients, and enough
potential forecast events. These are **optimistic timing bounds** from sidecar
durations, modality offsets, the window/stride grid, annotation timing and chosen
seizure prefixes. Signal quality, channel names, exclusion periods, feature
history, negative-class availability and actual grouped folds still require the
existing pipeline readiness checks. Header-based alignment or missing metadata
makes the timing preview unavailable; manual assignments remain possible.
If no suggestion is possible, the panel explains why without changing settings.

## Two-stage workflow and storage

The Configure page is deliberately split into two steps:

1. **Prepare data** contains the cohort, signal, alignment, preprocessing, label and
   feature settings. Choose **Feature extraction** to create a reusable prepared
   snapshot. Choose **Preprocessing only** when diagnosing waveform/label failures;
   it does not create a training-ready feature snapshot.
2. **Configure training** exposes only training and reporting settings. Select one
   completed prepared snapshot from its picker, then set the patient roles and run
   readiness or fitting. It never re-runs preprocessing or feature extraction.

Preparation writes the shared, immutable data snapshot to
`audit/runs/prepared/<preparation-job-id>/`. Training jobs reference that directory;
they do not copy it and do not change it. A snapshot fixes its raw-data folder,
selected recordings, modalities, alignment, preprocessing, labels and feature
configuration. Changing any of those requires a new preparation run. Training
settings and patient role assignments remain independently configurable.

This means an error after processing many patients can be debugged at the
preparation stage using its log and `qualification.json`, without repeating model
fitting. Once preparation succeeds, failed readiness or training runs can be
retried from exactly the same data snapshot.

Each preparation snapshot contains `preparation_config.yaml`, containing only the
data-preparation settings. Every dashboard job also stores that file and a separate
`training_config.yaml` containing only model, split and reporting settings. The
complete `config.yaml` remains available for reproducibility. In the dashboard, the
advanced JSON editor is available only in **Prepare data**; training hides it so a
prepared snapshot cannot be changed accidentally.

## Storage and isolation

Every launched experiment writes:

```text
audit/runs/dashboard/<job-id>/
  config.yaml          # immutable experiment configuration
  state.json           # persistent stage statuses
  pipeline.log         # complete subprocess output
  training/<run>/      # trained model and validation artifacts
audit/runs/prepared/<job-id>/
  ...                  # immutable, shared preprocessing/features snapshot
```

Raw data remains read-only to this workflow. The dashboard assigns output folders automatically and requires the raw directory to be inside this project. Its default is `data/raw`. Prepared snapshots are shared only through an explicit selection and configuration-compatibility check; unrelated jobs remain isolated. Existing model hyperparameter grids still run inside each training experiment.

One experiment runs at a time. Cancellation terminates the active Windows process tree, including parallel fit workers, and leaves partial artifacts for inspection. A cancelled/failed run is not presented as complete. The Windows job handle also kills its owned children when the server process closes. On restart, unfinished jobs are marked interrupted; they are not silently resumed.

## Architecture and API

- `-m src.commands.dashboard`: launcher.
- `src/dashboard/server.py`: Python standard-library HTTP server, loopback binding, JSON API and static assets.
- `src/dashboard/jobs.py`: job snapshots, isolated paths, stage subprocesses, logs, cancellation and artifact discovery.
- `src/dashboard/windows_process.py`: Windows process-tree ownership.
- `src/dashboard/static/`: framework-free browser UI; all assets are local.

Endpoints:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/bootstrap` | Default configuration, downloaded patient list and session token |
| POST | `/api/config/validate` | Validate a draft configuration |
| POST | `/api/patients` | Fast patient discovery and required-file coverage from BIDS filenames |
| POST | `/api/patients/details` | Optional sidecar/event timing and potential-event preview |
| POST | `/api/patients/suggest` | Preview a reproducible, disjoint patient split without modifying settings or launching a job |
| POST | `/api/jobs` | Start a named experiment with `config`, `mode`, `name` |
| POST | `/api/jobs` with `prepared_job` | Run only readiness/training using a compatible completed feature snapshot |
| GET | `/api/jobs/<id>/prepared-config` | Load configuration with the source's exact prepared cohort frozen |
| POST | `/api/jobs/<id>/repair-split` | Preview a repaired split from actual prepared labels; no fitting or source mutation |
| GET | `/api/jobs` | Experiment history |
| GET | `/api/jobs/<id>` | Stage status |
| GET | `/api/jobs/<id>/log` | Most recent 100 KB of log output |
| POST | `/api/jobs/<id>/cancel` | Cancel the active experiment |
| GET | `/api/jobs/<id>/config` | Saved configuration |
| GET | `/api/jobs/<id>/results` | Metrics and available artifacts |
| GET | `/api/jobs/<id>/files/<artifact>` | Allowlisted report/config/log file |

Mutations require JSON and the `X-Dashboard-Token` returned by bootstrap. The service accepts local Host/Origin values and sends no CORS permissions. Artifacts are served from explicit job-relative allowlists. This is a local single-user research application, not an authenticated multi-user or publicly hosted service.

## Verification

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
# With dashboard running and python -m src.tests.smoke.training's synthetic fixtures available:
.venv/Scripts/python.exe -m src.tests.smoke.dashboard
```

Backend tests cover configuration validation, immutable snapshots, stage sequencing, failure propagation, cancellation, concurrent-launch rejection, origin/token checks, path restrictions and JSON artifact serving. The live smoke test executes the existing synthetic EDF cohort through the HTTP API, verifies both evaluation partitions and fetches a rendered figure. Synthetic metrics are engineering checks, not clinical efficacy results.

### Completed verification

- All 38 automated tests passed, including rejection of empty patient partitions before processing.
- Full multimodal synthetic experiment completed through the API, including training and both validation/test reports.
- A second full EEG-only experiment completed in approximately 23 seconds, confirming parameter changes propagate through preparation and training.
- A separate EEG-only preprocessing experiment was launched and completed through the browser controls.
- The browser displayed stage logs, metric cards, curves and report links. Defaults were restored in the editor after testing.

For a quick interactive demonstration, select **Synthetic EEG-only training test** in experiment history, click **Use these settings**, change a parameter, choose **Training + validation + test**, and start it. These saved settings use tiny synthetic recordings and small model grids. Select **Reset defaults** before configuring your real-data experiment.

The second API verification can be repeated with `.venv/Scripts/python.exe -m src.tests.smoke.dashboard --eeg-only`. Its report is saved in `audit/runs/dashboard/43b3e85d6fd84506b4edc419c02464e4/DASHBOARD_SMOKE.json` for the verification performed on 2026-09-16.
