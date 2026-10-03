"""Data responsibilities for the training pipeline."""
import json
from pathlib import Path
import pandas as pd
from src.data.bids_loader import BIDSLoader
from src.pipeline.qualification import settings, record


def qualified_entries(config, entries):
    """Use only subjects explicitly assigned to a training partition when enabled."""
    q=settings(config)
    if not q['enabled'] or not q['exclude_unassigned_subjects']:
        return entries
    assigned=set().union(*[set(v) for v in config['training']['splits'].values()])
    kept=[]
    for entry in entries:
        if entry['subject'] in assigned:
            kept.append(entry)
        else:
            record(config, scope='subject', identifier=entry['subject'],
                   reason='not_assigned_to_training_partition', eligible_windows=entry.get('eligible'))
    return kept


def load_prepared_metadata(config):
    """Read existing window labels for split diagnostics, without EDFs or feature matrices."""
    entries=json.loads((Path(config['dataset']['interim_dir'])/'feature_manifest.json').read_text())
    expected={r['id'] for r in BIDSLoader(config['dataset']['raw_dir']).selected_runs(config['dataset'])}
    if not {e['run'] for e in entries}.issubset(expected):
        raise ValueError('Prepared recordings differ from the selected/downloaded recordings; prepare the updated cohort first')
    entries=qualified_entries(config,entries)
    frames=[]
    for entry in entries:
        for section in ('features','preprocessing','alignment','sensors','labels'):
            if entry['config'][section]!=config[section]:
                raise ValueError(f'Stale features: {section} differs; rebuild features')
        w=pd.read_csv(entry['windows_file'],usecols=['subject','label','event_row','prediction_time_sec','eligible'])
        w=w.loc[w.eligible].copy()
        w['recording_id']=entry['run']
        frames.append(w)
    if not frames:
        raise ValueError('No prepared recordings')
    return pd.concat(frames,ignore_index=True)


def load_feature_data(config):
    path=Path(config['dataset']['interim_dir'])/'feature_manifest.json'
    entries=json.loads(path.read_text())
    expected={r['id'] for r in BIDSLoader(config['dataset']['raw_dir']).selected_runs(config['dataset'])}
    if not {e['run'] for e in entries}.issubset(expected):
        raise ValueError('Feature manifest does not match current downloaded/selected recordings; prepare them first')
    entries=qualified_entries(config,entries)
    # The feature frames are filtered to eligible rows below.  Report both this
    # actual dense matrix estimate and the prepared total so a user can decide
    # whether a smaller patient-role subset can reuse the same snapshot.
    dictionaries=[json.loads((Path(e['feature_file']).parent/'feature_dictionary.json').read_text()) for e in entries]
    feature_names=set().union(*(set(d) for d in dictionaries))
    prepared_rows=sum(e['rows'] for e in entries)
    eligible_rows=sum(e.get('eligible',e['rows']) for e in entries)
    estimate=eligible_rows*len(feature_names)*8/1024**3
    prepared_estimate=prepared_rows*len(feature_names)*8/1024**3
    if estimate>config['training']['max_feature_matrix_gib']:
        raise ValueError(
            f'Estimated usable feature matrix {estimate:.2f} GiB ({eligible_rows:,} eligible rows × '
            f'{len(feature_names)} features, dense float64) exceeds configured limit '
            f"{config['training']['max_feature_matrix_gib']:.2f} GiB. The snapshot has {prepared_rows:,} prepared "
            f'rows ({prepared_estimate:.2f} GiB before eligibility filtering). This excludes DataFrame, feature-selection '
            'and cross-validation working copies, so do not merely raise the limit to this number. Reuse the snapshot '
            'with fewer assigned training/validation/test patients, reduce extracted features, or use a higher-memory '
            'machine/streaming trainer.')
    frames=[];metadata=[];events={}
    for e in entries:
        for section in ('features','preprocessing','alignment','sensors','labels'):
            if e['config'][section]!=config[section]:
                raise ValueError(f'Stale features: {section} differs; rebuild features')
        # These pickle files are locally generated artifacts. Never load untrusted pickle files.
        x=pd.read_pickle(e['feature_file'])
        w=pd.read_csv(e['windows_file'])
        if len(x)!=len(w):
            raise ValueError('Feature/metadata row mismatch')
        use=w.eligible.astype(bool)
        w=w.loc[use].copy();x=x.loc[use].copy()
        w['recording_id']=e['run']
        frames.append(x);metadata.append(w)
        events[e['run']]=pd.read_csv(e['events_file'],sep='\t')
    if not frames:
        raise ValueError('No prepared feature data')
    # Union preserves named missing channels as NaN, never shifts column positions.
    X=pd.concat(frames,ignore_index=True,sort=False)
    meta=pd.concat(metadata,ignore_index=True)
    return X,meta,events,entries


