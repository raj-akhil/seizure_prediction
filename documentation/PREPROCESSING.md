# Preprocessing and data contract

Run from the project root with `python -m src.commands.preprocess`. See the [project guide](../README.md) for all commands.

## Change an experiment

- `sensors.enabled: [ecg]` reads only ECG signals (EEG header remains the event time reference).
- `[eeg, ecg]` selects both; `[eeg, ecg, mov]` includes motion.
- `sensors.channels.eeg: null` selects every available EEG channel. Exact lists select named channels in the specified order. Missing channels fail or skip the entire run, according to `missing_policy`.
- Patient 125 EEG names include `BTEright SD` and `CROSStop SD`. A cross-head channel is not an independent left-ear channel.
- `output_sampling_rate: 256` creates aligned arrays at 256 Hz, including resampled motion. This does not add information to the original 25 Hz motion signal. `null` preserves native rates in separate arrays.
- `window_sec`, `stride_sec`, and `context_sec` control signal input duration and spacing. Total input duration is window plus context.
- Filter cutoffs and order are configured separately for each modality. Butterworth filtering is causal. Each window reads additional past `filter_warmup_sec` data; no samples after prediction time are used. The additional support is recorded and included in exclusion checks. This is finite warm-up filtering, not a persistent online filter state.
- Resampling uses polyphase anti-alias filtering within observed windows. Boundary effects can occur; no waveform after forecast time is read.
- Durations and offsets must fall exactly on each selected input/output sample grid. Unsupported fractional offsets fail rather than rounding silently.
- `seizure_prefixes: [sz]` includes all seizure categories. `[sz_foc]` restricts targets to focal categories, including focal-to-bilateral events. Other seizures remain excluded, not relabelled as background.

## Timing: an explicit assumption, not a verified clock correction

EEG/ECG/EMG headers in the inspected pilot use a 2000 placeholder date while motion uses a 2025 date. Therefore the supplied configuration uses `published_run_relative`: sample zero is assumed to refer to the publisher-aligned run origin, with optional corrections. Evidence is recorded in every report. See the SeizeIT2 publication: https://doi.org/10.1038/s41597-025-05580-x.

This is **not independent signal-level verification of synchronization**. `header` mode instead uses EDF timestamp differences and rejects the present non-overlapping timestamps. There is no automatic drift estimation or correction. EDF annotations currently trigger an error for review, so gaps/artifacts cannot be silently ignored.

`alignment.offsets_sec` sets per-modality offsets relative to the EEG/event origin: positive means that modality starts later. `alignment.run_offsets_sec` can override these using a full ID, for example `sub-125_ses-01_task-szMonitoring_run-21`. Only the common covered interval is used. Runs are never concatenated; their chronological continuity is unknown.

## Prediction labels

At prediction time t (the end of the observed input), target a seizure onset in:

`[t + minimum_warning_min, t + minimum_warning_min + occurrence_period_min)`

Times in that expression are converted to seconds internally. Defaults mean **30–40 minutes ahead**, not an exact 30-minute event time.

- `1`: selected seizure in the future interval.
- `0`: observed background satisfying exclusion rules.
- `-1`: excluded/unknown; never train on these rows.

The whole forecast interval must be observed, even for a positive example. Missing event files and unknown event codes fail. `annotation_coverage: whole_run` explicitly assumes the dataset's event lists cover the entire run; omitted time is background, not unreviewed time. Verify this convention for any new dataset.

Input overlapping seizures, postictal time, or impedance checks is excluded. A seizure in the minimum-warning gap is excluded. Unselected seizures in the forecast interval are excluded. Cluster conflicts cannot overwrite exclusions with positive labels. Original event rows and their subtype/lateralization/localization/vigilance metadata are preserved for later subgroup analysis. These annotations are not signal features.

## Quality and output contract

Nonfinite data is invalid. Configured electrical modalities reject windows containing a completely flat channel; constant motion axes are allowed. This is basic screening, **not comprehensive artifact removal**. This waveform stage does not perform ICA, learned cleaning, HR/HRV extraction or fitted scaling. Cardiac features are extracted in the separate feature stage; fitted transforms are learned within training folds.

`audit/runs/cli_prepared/latest_run.json` lists the latest selected runs and their output locations.

- `signals/<hash>/<run>/`: one float32 `.npy` array per selected modality, shaped **windows x channels x samples**; `quality_valid.npy`; `complete.json` with channel names, original units, calibrated-array units, rates, header dates, offsets, library versions and alignment evidence.
- `experiments/<hash>/<run>/`: `windows.csv`, `y.npy`, original `events.tsv`, configuration snapshot and report. Window rows include input start, processing support start, prediction time, eligibility, exclusion reason and matching event row.

The same window index across modality arrays denotes the same run-relative interval. Signal values use MNE calibration: electrical signals are in volts; motion follows its EDF calibration. No cross-modality unit normalization is performed. Invalid windows are retained and marked; filter by eligibility before model use.

`src.data.prepared.PreparedRun` memory-maps outputs. `get(index)` returns separate modality arrays and row metadata; `get(index, combined=True)` concatenates channels when rates match. `channel_names` contains modality-qualified channel names. The reader selects eligible windows by default. Call `close()` after consuming arrays; do not use returned mapped views afterward.

## Efficiency and reproducibility

Only selected signals are read, in bounded windows with warm-up, and arrays are written through memory maps. This bounds RAM usage but overlapping windows reread some data. No whole-run preload or giant flattened matrix is created.

Completed signal caches are keyed by source paths/sizes/modification times, sidecars, selected sensors, alignment, preprocessing, implementation and library versions. Changing label parameters reuses signals and creates separate labels. Changing preprocessing creates a new cache. Source fingerprints are metadata-based, not full EDF content hashes; preserve source immutability. Incomplete caches lack a completion marker and are rebuilt. Do not run concurrent writers to the same cache.

Tests cover label boundaries, seizure clusters, missing annotations/channels, incompatible timestamps, ECG-only and multimodal processing, changed window/context/stride, manual offsets, native rates, and label-only cache reuse. Patient-dependent tests skip if patient 125 is absent.

