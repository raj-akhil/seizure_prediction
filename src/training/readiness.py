"""Explain and repair patient partitions using actual prepared labels, without fitting."""
import copy
import itertools
import random

import pandas as pd

from .splits import preflight

PARTITIONS = ('train', 'validation', 'test')


def counts(meta):
    positive = meta.loc[meta.label == 1]
    return {'windows': len(meta), 'background_windows': int((meta.label == 0).sum()),
            'positive_windows': len(positive),
            'positive_events': len(positive[['recording_id', 'event_row']].drop_duplicates())}


def report(config, meta):
    """Use the same preflight as training, including the actual grouped CV folds."""
    result = {'ready': False, 'forecast_minutes': [config['labels']['minimum_warning_min'],
              config['labels']['minimum_warning_min'] + config['labels']['occurrence_period_min']],
              'minimum_events_per_partition': config['training']['min_positive_events_per_partition'],
              'inner_folds_required': config['training']['inner_folds'],
              'patients': [{'subject': subject, **counts(rows)} for subject, rows in meta.groupby('subject')],
              'partitions': {name: {'subjects': config['training']['splits'][name],
                 **counts(meta.loc[meta.subject.isin(config['training']['splits'][name])])} for name in PARTITIONS},
              'issues': [], 'inner_folds': []}
    try:
        # Preflight needs only row indices, labels and groups. No feature values or fitting.
        indices, folds = preflight(config, pd.DataFrame(index=meta.index), meta)
        train = meta.iloc[indices['train']]
        result['inner_folds'] = [{'train_subjects': sorted(train.iloc[a].subject.unique()),
                                 'validation_subjects': sorted(train.iloc[b].subject.unique()),
                                 'train': counts(train.iloc[a]), 'validation': counts(train.iloc[b])}
                                for a, b in folds]
        result['ready'] = True
    except ValueError as exc:
        result['issues'] = [str(exc)]
    return result


def repair_splits(config, meta):
    """Find a small change that passes the existing safeguards; never relax them.

Exhaustive assignments for up to nine patients; bounded seeded search for larger
cohorts. Labels are used for partition feasibility, never model/test performance.
    """
    current = report(config, meta)
    if current['ready']:
        return {'config': copy.deepcopy(config), 'before': current, 'after': current, 'changed': False}
    patients = current['patients']
    minimum, folds = config['training']['min_positive_events_per_partition'], config['training']['inner_folds']
    if len(patients) < folds + 2:
        raise ValueError(f'Cannot repair with unchanged settings: need at least {folds + 2} eligible patients.')
    if sum(row['positive_events'] for row in patients) < minimum * 3:
        raise ValueError('Cannot repair with unchanged settings: insufficient eligible seizure episodes across the cohort.')
    original = {row['subject']: {i for i, name in enumerate(PARTITIONS)
                               if row['subject'] in config['training']['splits'][name]} for row in patients}
    rng = random.Random(config['training']['seed'])
    if len(patients) <= 9:
        assignments = itertools.product(range(3), repeat=len(patients))
    else:
        assignments = (tuple(rng.choices(range(3), weights=[.7, .15, .15], k=len(patients))) for _ in range(4096))
    options = []
    for assignment in assignments:
        groups = [[row for row, role in zip(patients, assignment) if role == i] for i in range(3)]
        if len(groups[0]) < folds or any(not group for group in groups):
            continue
        if any(sum(row['positive_events'] for row in group) < minimum
               or not sum(row['background_windows'] for row in group) for group in groups):
            continue
        if sum(row['positive_windows'] > 0 for row in groups[0]) < folds:
            continue
        if len(groups[0]) == folds and any(not row['background_windows'] for row in groups[0]):
            continue
        distance = sum(original[row['subject']] != {role} for row, role in zip(patients, assignment))
        options.append((distance, assignment))
    options.sort()
    for _, assignment in options[:128]:
        candidate = copy.deepcopy(config)
        candidate['training']['splits'] = {name: [row['subject'] for row, role in zip(patients, assignment) if role == i]
                                           for i, name in enumerate(PARTITIONS)}
        checked = report(candidate, meta)
        if checked['ready']:
            return {'config': candidate, 'before': current, 'after': checked, 'changed': True}
    raise ValueError('No valid repair found with unchanged event minimums and grouped folds. '
                     'Add eligible data or explicitly reconsider the study settings; safeguards were retained.')
