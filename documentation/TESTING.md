# Tests and verification

Test code lives under `src/tests/`; historical notes are in `documentation/history/testing/`. Generated test data and results live under `audit/runs/`, never beside test source files.

| Location | Purpose |
|---|---|
| `test_*.py` | Automated regression tests, including temporary tiny fixtures |
| `smoke/training.py` | Small synthetic EDF-to-training workflow and a bounded real-feature compatibility check |
| `smoke/notebooks.py` | Execute both notebooks in separate fresh kernels and verify saved predictions/report compatibility |
| `smoke/dashboard.py` | Exercise the live dashboard API using previously generated tiny synthetic fixtures |
| `smoke/real_data.py` | Bounded check on four existing real recordings, with explicit splits and separate pilot settings |
| `checks/preprocessing.py` | Opt-in audit of prepared waveform artifacts |
| `checks/features.py` | Opt-in audit of extracted feature artifacts |

Run commands from the project root:

```powershell
# Normal fast regression suite
.venv/Scripts/python.exe -m unittest discover -s src/tests -v

# Small end-to-end tests (fit synthetic models)
.venv/Scripts/python.exe -m src.tests.smoke.training
.venv/Scripts/python.exe -m src.tests.smoke.notebooks
.venv/Scripts/python.exe -m src.tests.smoke.notebooks --real
.venv/Scripts/python.exe -m src.tests.smoke.dashboard --eeg-only
.venv/Scripts/python.exe -m src.tests.smoke.real_data --timeout-sec 180

# Optional data audits: these inspect every artifact in the selected manifest
.venv/Scripts/python.exe -m src.tests.checks.preprocessing --config src/config/config.yaml
.venv/Scripts/python.exe -m src.tests.checks.features --config src/config/config.yaml
```

Dashboard smoke tests require `python -m src.commands.dashboard` running and synthetic fixtures created by the training smoke test. The audit commands are separate from automatic test discovery because real-data audits may take longer. Use a small saved experiment configuration to audit a small cohort. Optional `--output-dir` controls where audit reports are written; defaults are `audit/runs/reports/preprocessing` and `audit/runs/reports/features`.

Synthetic scores demonstrate software behavior only. They do not establish real seizure-forecasting performance.

Notebook checks need `requirements.txt`. The default creates four tiny synthetic multisensor participants. `--real` selects four exact recordings from local participants 001, 002, 004 and 007, using EEG only, a 120-second stride, two grouped folds and the original 30–40 minute target. It stops if those sources are absent. These bounded checks do not run the entire ten-participant cohort. Executed notebook copies and results stay in `audit/runs/smoke_tests/notebooks/`; the authored notebooks retain empty outputs. Both checks use a separate dataset name and preserve existing handoffs.
