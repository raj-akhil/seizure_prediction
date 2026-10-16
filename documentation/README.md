# Documentation

Start with the [project README](../README.md) and the two notebooks in the project root.

| Guide | Purpose |
|---|---|
| [Preparation notebook](../01_prepare_data.ipynb) | Parameters, source inventory, signals, labels, features and debugging |
| [Training notebook](../02_train_model.ipynb) | Patient splits, fitting, threshold selection, evaluation and saving |
| [Preprocessing](PREPROCESSING.md) | Signal/label definitions, timing assumptions and file contracts |
| [Training](TRAINING_PIPELINE.md) | Feature selection, models, patient splits and the optional LSTM path |
| [Validation](VALIDATION_PIPELINE.md) | Metrics, uncertainty and research limitations |
| [Research decisions](RESEARCH_DECISIONS.md) | Feature/model choices and rationale |
| [Audit and outputs](AUDIT.md) | Prepared data, logs, models and reports |
| [Testing](TESTING.md) | Regression checks and bounded notebook execution |
| [Dashboard](DASHBOARD.md) | Optional older interface: `python -m src.commands.dashboard` |
| [Historical real-data check](REAL_DATA_CHECK.md) | Earlier pilot; its patients differ from the current example |

Historical testing notes are in `history/testing/`. The reference presentation is in `references/`. Generated PDFs/presentations are under `../audit/archive/output/`; experiment reports live under `../audit/runs/`.
