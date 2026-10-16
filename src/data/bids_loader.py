import re
from pathlib import Path


class BIDSLoader:
    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)

    def discover_runs(self):
        runs = {}
        pattern = re.compile(r'(?P<id>(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-[^_]+_(?P<run>run-[^_]+))_(?P<mod>eeg|ecg|emg|mov)\.edf$', re.I)
        for p in sorted(self.raw_dir.rglob('*.edf')):
            m = pattern.fullmatch(p.name)
            if not m:
                continue
            d = m.groupdict()
            r = runs.setdefault(d['id'], {k:d[k] for k in ('id','subject','session','run')} | {'files':{}})
            mod = d['mod'].lower()
            if mod in r['files']:
                raise ValueError(f'Duplicate modality for {d["id"]}')
            r['files'][mod] = p
            event = p.parent.parent / 'eeg' / (d['id'] + '_events.tsv')
            r['files']['events'] = event if event.exists() else None
        return list(runs.values())

    def selected_runs(self, dataset):
        """One selection rule shared by preparation, features and training."""
        runs=[r for r in self.discover_runs()
              if (dataset.get('subjects') is None or r['subject'] in dataset['subjects'])
              and (dataset.get('runs') is None or r['run'] in dataset['runs'])
              and (dataset.get('recordings') is None or r['id'] in dataset['recordings'])]
        if dataset.get('recordings') is not None:
            missing=set(dataset['recordings'])-{r['id'] for r in runs}
            if missing:
                raise ValueError(f'Recording IDs missing or excluded by patient/run filters: {sorted(missing)}')
        return runs


def require_complete_recordings(runs, config):
    """Inspect paths before expensive signal reads; never silently drop a recording."""
    required=set(config['sensors']['enabled'])|{'eeg','events'}
    missing=[f"{r['id']}: {', '.join(sorted(m for m in required if not r['files'].get(m)))}"
             for r in runs if any(not r['files'].get(m) for m in required)]
    if missing and config['sensors']['missing_policy']=='error':
        raise ValueError(f'{len(missing)} selected recordings have missing files. '
                         + '; '.join(missing[:8])
                         + '. Finish copying these recordings, choose available modalities, or explicitly select complete recording IDs.')
