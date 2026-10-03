# Generated outputs

All generated run reports, figures, logs and trained model artifacts belong here.

| Directory | Contents |
|---|---|
| `dashboard/<job-id>/` | Each dashboard job's immutable config, logs, readiness output and trained-model reports |
| `prepared/<job-id>/` | Shared immutable preprocessing and feature snapshot; selected explicitly by later training jobs |
| `training/<run-id>/` | Models, predictions and reports from command-line training |
| `smoke_tests/<timestamp>/` | Synthetic test recordings, temporary prepared data, models and test summaries |
| `reports/preprocessing/` | Explicit preprocessing audit summaries (JSON/CSV/Markdown) |
| `reports/features/` | Explicit feature audit summaries |
| `figures/` | Standalone exported figures and PDFs, when produced outside an experiment |
| `logs/` | Standalone operational logs, when produced outside an experiment |

Figures and reports tied to an experiment remain inside that experiment folder so their configuration and provenance stay together. The project currently generates PNG, Markdown, JSON and CSV reports; there is no PDF-generation stage.

Existing experiment directories have not been moved: stored manifests contain absolute artifact paths. Do not move individual files out of these directories without updating their consumers. Raw patient recordings remain in `data/raw/`, and reusable CLI preprocessing arrays/features remain in `data/interim/`.
