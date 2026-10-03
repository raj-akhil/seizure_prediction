# Tests and verification

All test code and historical test notes live under this directory. Generated test data and results live under `outputs/`, never beside test source files.

| Location | Purpose |
|---|---|
| `test_*.py` | Automated regression tests, including temporary tiny fixtures |
| `smoke/training.py` | Small synthetic EDF-to-training workflow and a bounded real-feature compatibility check |
| `smoke/dashboard.py` | Exercise the live dashboard API using previously generated tiny synthetic fixtures |
| `smoke/real_data.py` | Bounded check on four existing real recordings, with explicit splits and separate pilot settings |
| `checks/preprocessing.py` | Opt-in audit of prepared waveform artifacts |
| `checks/features.py` | Opt-in audit of extracted feature artifacts |
| `docs/` | Historical test status notes |

Run commands from the project root:

```powershell
# Normal fast regression suite
.venv/Scripts/python.exe -m unittest discover -s tests -v

# Small end-to-end tests (fit synthetic models)
.venv/Scripts/python.exe -m tests.smoke.training
.venv/Scripts/python.exe -m tests.smoke.dashboard --eeg-only
.venv/Scripts/python.exe -m tests.smoke.real_data --timeout-sec 180

# Optional data audits: these inspect every artifact in the selected manifest
.venv/Scripts/python.exe -m tests.checks.preprocessing --config config/config.yaml
.venv/Scripts/python.exe -m tests.checks.features --config config/config.yaml
```

Dashboard smoke tests require `dashboard.py` running and synthetic fixtures created by the training smoke test. The audit commands are separate from automatic test discovery because real-data audits may take longer. Use a small saved experiment configuration to audit a small cohort. Optional `--output-dir` controls where audit reports are written; defaults are `outputs/reports/preprocessing` and `outputs/reports/features`.

Synthetic scores demonstrate software behavior only. They do not establish real seizure-forecasting performance.
