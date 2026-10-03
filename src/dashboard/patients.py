"""Fast, read-only patient discovery and advisory partition suggestions.

Only filenames, JSON sidecars and event tables are read. Potential event counts
are optimistic timing bounds, NOT feature/training readiness or signal QC.
"""
import json
import math
import random
from pathlib import Path

from src.data.bids_loader import BIDSLoader
from src.preprocessing.labels import read_events


PARTITIONS = ('train', 'validation', 'test')


def _duration(path):
    value = float(json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))['RecordingDuration'])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid RecordingDuration')
    return value


def potential_events(run, config):
    """Upper bound from the decision grid, common interval and future coverage.

Ignore exclusion periods, additional feature history and signal quality. These
can only remove candidate events. Never read EDF waveforms in this preview.
    """
    if config['alignment']['mode'] != 'published_run_relative':
        raise ValueError('Timing preview needs published run-relative alignment; header alignment is checked during preprocessing')
    mods = config['sensors']['enabled']
    durations = {mod: _duration(run['files'][mod]) for mod in set(mods) | {'eeg'}}
    events = read_events(run['files']['events'], durations['eeg'])
    offsets = dict(config['alignment']['offsets_sec'])
    offsets.update(config['alignment'].get('run_offsets_sec', {}).get(run['id'], {}))
    if any(not math.isfinite(float(offsets.get(mod, 0))) for mod in mods):
        raise ValueError('Invalid modality time offset')
    start = max(0, *(offsets.get(mod, 0) for mod in mods))
    end = min(durations['eeg'], *(offsets.get(mod, 0) + durations[mod] for mod in mods))
    p, labels = config['preprocessing'], config['labels']
    first = start + p['filter_warmup_sec'] + p['window_sec'] + p['context_sec']
    horizon, period = labels['minimum_warning_min'] * 60, labels['occurrence_period_min'] * 60
    last = min(end, durations['eeg'] - horizon - period)
    stride = p['stride_sec']
    count = 0
    for _, event in events.iterrows():
        if not any(event.eventType.startswith(prefix) for prefix in labels['seizure_prefixes']):
            continue
        # event onset in [decision + horizon, decision + horizon + period).
        index = max(0, math.floor((event.onset - horizon - period - first) / stride) + 1)
        decision = first + index * stride
        if decision <= last + 1e-8 and decision + horizon <= event.onset < decision + horizon + period:
            count += 1
    return count


def patient_catalog(raw_dir, config, include_timing=True):
    """List the available patient data.

    The default includes the advisory event/timing preview used by split
    suggestions.  Callers that only need a cohort list can opt out: discovery
    then reads filenames only, so a large downloaded cohort becomes visible
    without waiting for every sidecar and annotation table.
    """
    raw = Path(raw_dir).resolve()
    runs = BIDSLoader(raw).discover_runs()
    dataset = config['dataset']
    splits = config['training']['splits']
    subjects = {p.name for p in raw.glob('sub-*') if p.is_dir()}
    subjects.update(run['subject'] for run in runs)
    subjects.update(dataset.get('subjects') or [])
    subjects.update(subject for group in splits.values() for subject in group)
    required = set(config['sensors']['enabled']) | {'eeg', 'events'}
    filtered = [run for run in runs
                if (dataset.get('runs') is None or run['run'] in dataset['runs'])
                and (dataset.get('recordings') is None or run['id'] in dataset['recordings'])]
    rows = []
    for subject in sorted(subjects):
        chosen = [run for run in filtered if run['subject'] == subject]
        issues, incomplete, potential, unknown = [], 0, 0, not include_timing
        for run in chosen:
            missing = sorted(mod for mod in required if not run['files'].get(mod))
            if missing:
                incomplete += 1
                unknown = True
                issues.append(f"{run['run']}: missing {', '.join(missing)}")
                continue
            if not include_timing:
                continue
            try:
                # Prevent metadata reads through links outside the configured raw folder.
                for path in run['files'].values():
                    if path and (not path.resolve().is_relative_to(raw)
                                 or not path.with_suffix('.json').resolve().is_relative_to(raw)):
                        raise ValueError('File links outside the raw folder are not supported')
                potential += potential_events(run, config)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                unknown = True
                issues.append(f"{run['run']}: metadata preview unavailable ({exc})")
        rows.append({'id': subject, 'recordings': len(chosen),
                     'total_recordings': sum(run['subject'] == subject for run in runs),
                     'incomplete_recordings': incomplete,
                     'potential_events': None if unknown else potential,
                     'issues': issues[:3], 'issue_count': len(issues),
                     'selected': dataset.get('subjects') is None or subject in dataset['subjects']})
    selected_ids = {run['id'] for run in filtered
                    if dataset.get('subjects') is None or run['subject'] in dataset['subjects']}
    missing_ids = sorted(set(dataset.get('recordings') or []) - selected_ids)
    return {'patients': rows, 'raw_dir': str(raw),
            'timing_checked': include_timing,
            'warnings': [f'Recordings missing or excluded by patient/run filters: {", ".join(missing_ids)}'] if missing_ids else [],
            'note': ('Potential forecast events are an optimistic estimate from JSON durations and annotation timing. '
                     'Signal quality, channel availability, exclusion periods, feature history and class balance are checked by the pipeline. '
                     'No EDF samples are loaded.' if include_timing else
                     'Fast listing reads BIDS filenames only. Use “Check timing & event details” before reviewing event counts or requesting a suggested split.')}


def suggest_split(catalog, config):
    """Reproducible patient-level draft; never changes cohort or model settings."""
    patients = [row for row in catalog['patients'] if row['selected']]
    folds = config['training']['inner_folds']
    minimum = config['training']['min_positive_events_per_partition']
    reasons = list(catalog['warnings'])
    for row in patients:
        if not row['recordings']:
            reasons.append(f"{row['id']}: no recordings match the current run/recording filters.")
        elif row['incomplete_recordings']:
            reasons.append(f"{row['id']}: {row['incomplete_recordings']} recordings have missing required files. Finish copying or adjust your selection.")
        elif row['potential_events'] is None:
            reasons.append(f"{row['id']}: timing metadata could not be checked; review the patient details.")
    if len(patients) < folds + 2:
        reasons.append(f'Need at least {folds + 2} selected patients: {folds} for grouped training folds and one each for validation and test; found {len(patients)}.')
    known = [row for row in patients if row['potential_events'] is not None]
    if len(known) == len(patients):
        positives = sum(row['potential_events'] > 0 for row in known)
        if positives < folds + 2:
            reasons.append(f'Only {positives} patients have potential forecast events at this horizon. At least {folds + 2} are needed to supply positive examples to every training fold, validation and test.')
        if sum(row['potential_events'] for row in known) < 3 * minimum:
            reasons.append(f'Not enough potential events for at least {minimum} in each partition. Add data or explicitly revise the forecast/event settings.')
    if reasons:
        return {'splits': None, 'reasons': reasons}
    # Approximately 70/15/15, with the configured training-fold minimum respected.
    train_size = min(len(patients) - 2, max(folds, math.floor(len(patients) * .7)))
    val_size = max(1, (len(patients) - train_size) // 2)
    rng = random.Random(config['training']['seed'])
    for _ in range(512):
        rng.shuffle(patients)
        groups = [patients[:train_size], patients[train_size:train_size + val_size], patients[train_size + val_size:]]
        if any(sum(row['potential_events'] for row in group) < minimum for group in groups):
            continue
        if sum(row['potential_events'] > 0 for row in groups[0]) < folds:
            continue
        return {'splits': {name: sorted(row['id'] for row in group) for name, group in zip(PARTITIONS, groups)},
                'reasons': [], 'note': 'Suggested draft using the training seed, roughly 70/15/15 where cohort size allows. '
                'Patient groups do not overlap. Review before applying; final class balance and grouped folds require the pipeline readiness check. '
                'Keep the test patients fixed when comparing experiments.'}
    return {'splits': None, 'reasons': ['No suitable draft found with these patient counts and event minimums. Assign manually or add patients; settings were not changed.']}
