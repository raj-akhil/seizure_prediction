import numpy as np
import pandas as pd


def read_events(path, duration):
    if path is None:
        raise ValueError('Missing event annotations; cannot assume background')
    events = pd.read_csv(path, sep='\t')
    if events.empty or not {'onset', 'duration', 'eventType'}.issubset(events.columns):
        raise ValueError('Invalid/empty event annotations')
    for _, e in events.iterrows():
        if not np.isfinite(e.onset) or not np.isfinite(e.duration) or e.onset < 0 or e.duration <= 0 or e.onset + e.duration > duration + .1:
            raise ValueError('Event outside recording or invalid duration')
        if not isinstance(e.eventType, str) or not (e.eventType.startswith('sz') or e.eventType in ('bckg', 'impd')):
            raise ValueError(f'Unknown event type: {e.eventType}')
    return events


def label_windows(starts, ends, events, duration, cfg):
    """Input ends are forecast times. Future intervals are half-open; exclusions win."""
    y = np.full(len(ends), -1, dtype=np.int8)
    reasons = np.full(len(ends), 'unobserved_future', dtype=object)
    matched = np.full(len(ends), '', dtype=object)
    h, p = cfg['minimum_warning_min'] * 60, cfg['occurrence_period_min'] * 60
    post, sep = cfg['postictal_exclusion_min'] * 60, cfg['background_separation_min'] * 60
    for i, (a, b) in enumerate(zip(starts, ends)):
        if b + h + p > duration + 1e-8:
            continue
        y[i], reasons[i] = 0, 'background'
        for idx, e in events.iterrows():
            selected = e.eventType.startswith('sz') and any(e.eventType.startswith(prefix) for prefix in cfg['seizure_prefixes'])
            if selected and b + h <= e.onset < b + h + p:
                y[i], reasons[i], matched[i] = 1, 'preictal', str(idx)
        for _, e in events.iterrows():
            seizure = e.eventType.startswith('sz')
            if seizure and a < e.onset + e.duration + post and b > e.onset:
                y[i], reasons[i] = -1, 'ictal_or_postictal_input'
            elif seizure and b <= e.onset < b + h:
                y[i], reasons[i] = -1, 'seizure_inside_warning_gap'
            elif seizure and not any(e.eventType.startswith(prefix) for prefix in cfg['seizure_prefixes']) and b + h <= e.onset < b + h + p:
                y[i], reasons[i] = -1, 'unselected_seizure'
            elif seizure and y[i] == 0 and a < e.onset + e.duration + sep and b > e.onset - sep:
                y[i], reasons[i] = -1, 'background_separation'
            elif e.eventType in cfg['excluded_event_types'] and a < e.onset + e.duration and b > e.onset:
                y[i], reasons[i] = -1, 'excluded_event_input'
        if y[i] < 0:
            matched[i] = ''
    return y, reasons, matched
