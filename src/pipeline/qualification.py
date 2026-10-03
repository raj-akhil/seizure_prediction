"""Persist auditable exclusions made while preparing a training cohort."""
import json
from pathlib import Path


def settings(config):
    """Defaults keep old saved experiment configurations usable."""
    return {'enabled': True, 'exclude_unassigned_subjects': True,
            'continue_on_recording_error': True} | config.get('qualification', {})


def report_path(config):
    return Path(config['dataset']['interim_dir']) / 'qualification.json'


def record(config, *, scope, identifier, reason, detail=None, eligible_windows=None):
    """Append one exclusion and print it so it appears in the pipeline log."""
    path = report_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError:
        report = {'qualified': [], 'excluded': []}
    item = {'scope': scope, 'id': identifier, 'reason': reason}
    if detail:
        item['detail'] = detail
    if eligible_windows is not None:
        item['eligible_windows'] = int(eligible_windows)
    if item not in report['excluded']:
        report['excluded'].append(item)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'FILTERED {scope.upper()}: {identifier} | {reason}'
              + (f' | {detail}' if detail else ''), flush=True)


def qualify(config, identifier):
    path = report_path(config)
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError:
        report = {'qualified': [], 'excluded': []}
    if identifier not in report['qualified']:
        report['qualified'].append(identifier)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')

