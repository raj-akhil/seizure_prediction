# Audit results, logs and saved experiments

Start with the two notebooks in the project root. All generated work belongs here.

| Folder | Contents |
|---|---|
| `runs/notebooks/<dataset-name>/` | Prepared snapshots, logs, handoff and timestamped model experiments |
| `runs/smoke_tests/` | Synthetic fixtures, executed notebook copies and check results |
| `runs/reports/` | Signal/feature audit summaries |
| Other folders under `runs/` | Earlier dashboard, shared preparation and CLI experiments |
| `archive/output/` | Earlier generated PDFs and presentations |
| `archive/tmp/` | Earlier temporary builders, previews and build directories |
| `archive/visual_studio/` | Earlier Visual Studio workspace state, when present |
| `maintenance/` | Layout migration record and raw-data inventory |

Cleanup preserved original recordings in `../data/raw/` and updated saved experiment paths. Generated contents are ignored by Git; this guide is tracked.

Keep a prepared snapshot while a model or handoff references it. Manifests use absolute paths, so avoid moving individual signal/feature files. For another experiment, choose a new `DATASET_NAME` or rerun preparation.

See the [output guide](../documentation/AUDIT.md) and [testing guide](../documentation/TESTING.md).
