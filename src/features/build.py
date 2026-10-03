import json
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from src.preprocessing.stream_processor import digest, save_json, integer_samples
from src.preprocessing.labels import label_windows, read_events
from .signal_features import channel_features, cardiac_features, divide
from src.data.bids_loader import BIDSLoader
from src.pipeline.qualification import settings, record


def build_features(config):
    """Extract all windows, preserve exclusions. Fitted selection is deferred to training."""
    out=Path(config['dataset']['interim_dir'])
    reports=json.loads((out/'latest_run.json').read_text())
    if any(r['status']!='complete' for r in reports) and not settings(config)['enabled']:
        raise ValueError('Preprocessing manifest contains failed/skipped runs; review before feature preparation')
    expected={r['id'] for r in BIDSLoader(config['dataset']['raw_dir']).selected_runs(config['dataset'])}
    observed={r['run'] for r in reports}
    if (observed!=expected if not settings(config)['enabled'] else not observed.issubset(expected)):
        raise ValueError('Downloaded/selected runs differ from preprocessing manifest; run python -m scripts.preprocess first')
    entries=[]
    feature_cfg=config['features']
    code={str(p):p.read_text() for p in Path(__file__).parent.glob('*.py')}
    for r in (r for r in reports if r['status']=='complete'):
        directory=Path(r['result_directory'])
        previous=json.loads((directory/'config.json').read_text())
        for key in ('preprocessing','alignment','sensors','labels'):
            if previous[key]!=config[key]:
                raise ValueError(f'Stale preprocessing for {r["run"]}: {key}; run python -m scripts.preprocess first')
        if config['dataset']['subjects'] is not None and previous['dataset']['subjects']!=config['dataset']['subjects']:
            raise ValueError('Preprocessing subject selection differs; run python -m scripts.preprocess first')
        key=digest({'report':r,'features':feature_cfg,'code':code})
        dest=out/'features'/key/r['run']
        dest.mkdir(parents=True,exist_ok=True)
        marker=dest/'complete.json'
        if marker.exists():
            entry=json.loads(marker.read_text())
            if Path(entry['feature_file']).exists():
                if entry['eligible']:
                    entries.append(entry)
                else:
                    record(config, scope='recording', identifier=r['run'], reason='no_eligible_windows')
                print(f'Features cached: {r["run"]}',flush=True)
                continue
        windows=pd.read_csv(directory/'windows.csv')
        count=len(windows)
        arrays={mod:np.load(Path(r['signal_directory'])/f'{mod}.npy',mmap_mode='r') for mod in r['inventory']}
        columns={}
        for mod,arr in arrays.items():
            info=r['inventory'][mod]
            family=feature_cfg['families'][mod]
            fs=info['output_rate']
            for a in range(0,count,feature_cfg['batch_windows']):
                b=min(count,a+feature_cfg['batch_windows'])
                with np.errstate(invalid='ignore',divide='ignore'):
                    block=channel_features(arr[a:b],fs,info['native_rate'],family,feature_cfg['bands'][mod],feature_cfg['welch_segment_sec'])
                for name,values in block.items():
                    for j,ch in enumerate(info['channels']):
                        col=f'{mod}::{ch}::{name}'
                        columns.setdefault(col,np.full(count,np.nan))[a:b]=values[:,j]
            if 'correlation' in family:
                for i,left in enumerate(info['channels']):
                    for j in range(i+1,len(info['channels'])):
                        col=f'{mod}::{left}__{info["channels"][j]}::correlation'
                        values=np.full(count,np.nan)
                        for a in range(0,count,feature_cfg['batch_windows']):
                            b=min(count,a+feature_cfg['batch_windows'])
                            x,y=arr[a:b,i].astype(float),arr[a:b,j].astype(float)
                            x-=x.mean(axis=1,keepdims=True);y-=y.mean(axis=1,keepdims=True)
                            values[a:b]=divide((x*y).mean(axis=1),x.std(axis=1)*y.std(axis=1))
                        columns[col]=values
            if 'motion' in family:
                for stem,axes in [('EEG SD ACC',['X','Y','Z']),('ECGEMG SD ACC',['X','Y','Z']),
                                  ('EEG SD GYR',['A','B','C']),('ECGEMG SD GYR',['A','B','C'])]:
                    names=[f'{stem} {axis}' for axis in axes]
                    if not set(names).issubset(info['channels']):
                        continue
                    ids=[info['channels'].index(ch) for ch in names]
                    for a in range(0,count,feature_cfg['batch_windows']):
                        b=min(count,a+feature_cfg['batch_windows'])
                        magnitude=np.sqrt(np.sum(arr[a:b,ids].astype(float)**2,axis=1))
                        for name,values in {'magnitude_mean':magnitude.mean(axis=1),'magnitude_std':magnitude.std(axis=1),
                            'magnitude_derivative_rms':np.sqrt(np.mean(np.diff(magnitude,axis=1)**2,axis=1))*fs}.items():
                            columns.setdefault(f'mov::{stem}::{name}',np.full(count,np.nan))[a:b]=values
        # Longer ECG history comes directly from calibrated EDF, ending at forecast time.
        # This does not depend on upsampled motion, or join/reset-filtered five-second ECG windows.
        support=windows.processing_support_start_sec.to_numpy().copy()
        history_valid=np.ones(count,dtype=bool)
        if 'ecg' in arrays and 'cardiac' in feature_cfg['families']['ecg']:
            info=r['inventory']['ecg']
            raw=mne.io.read_raw_edf(info['file'],preload=False,verbose=False)
            raw.pick(info['channels'])
            ecfg=feature_cfg['ecg']
            for i,t in enumerate(windows.prediction_time_sec):
                a=t-ecfg['history_sec']
                support[i]=min(support[i],a)
                if a<info['offset_sec']:
                    history_valid[i]=False
                    continue
                data=raw.get_data(start=integer_samples(a-info['offset_sec'],raw.info['sfreq']),
                                  stop=integer_samples(t-info['offset_sec'],raw.info['sfreq']))
                for j,ch in enumerate(info['channels']):
                    values=cardiac_features(data[j],raw.info['sfreq'],ecfg)
                    for name,value in values.items():
                        columns.setdefault(f'ecg::{ch}::{name}',np.full(count,np.nan))[i]=value
            raw.close()
        feature_frame=pd.DataFrame(columns).replace([np.inf,-np.inf],np.nan)
        base_columns=list(feature_frame.columns)
        original_support=support.copy()
        # Trailing summaries are time-based and require every expected window (no gap bridging).
        stride=config['preprocessing']['stride_sec']
        for seconds in feature_cfg['temporal']['history_sec']:
            n=integer_samples(seconds,1/stride)
            if n<2:
                raise ValueError('Temporal history must contain at least two strides')
            shifted=feature_frame[base_columns].rolling(n,min_periods=n).mean()
            shifted.columns=[f'{col}::trailing_mean_{seconds}s' for col in shifted.columns]
            feature_frame=pd.concat([feature_frame,shifted],axis=1)
            history_valid[:n-1]=False
            history_valid[n-1:] &= np.convolve(windows.signal_valid.astype(int),np.ones(n,dtype=int),'valid')==n
            shifted_support=pd.Series(original_support).rolling(n,min_periods=n).min().to_numpy()
            support[n-1:]=np.minimum(support[n-1:],shifted_support[n-1:])
        events=read_events(directory/'events.tsv',r['inventory'].get('eeg',next(iter(r['inventory'].values())))['sidecar']['RecordingDuration'])
        # Event duration from the original EEG header, including ECG-only experiments.
        event_duration=float(events.recordingDuration.iloc[0]) if 'recordingDuration' in events else min(info['duration_sec'] for info in r['inventory'].values())
        y,reasons,matched=label_windows(support,windows.prediction_time_sec.to_numpy(),events,event_duration,config['labels'])
        invalid=~windows.signal_valid.to_numpy()
        y[invalid],reasons[invalid],matched[invalid]=-1,'signal_quality',''
        y[~history_valid],reasons[~history_valid],matched[~history_valid]=-1,'insufficient_feature_history',''
        windows['label']=y;windows['eligible']=y>=0;windows['reason']=reasons;windows['event_row']=matched
        windows['feature_support_start_sec']=support
        windows.to_csv(dest/'windows.csv',index=False)
        feature_frame.to_pickle(dest/'features.pkl')
        dictionary={col:{'modality':col.split('::')[0],'channel_or_group':col.split('::')[1],
            'feature':col.split('::')[2], 'units_note':'Derived from calibrated EDF units; spectral power in squared units; frequency Hz; cardiac names specify ms or bpm'} for col in feature_frame}
        save_json(dest/'feature_dictionary.json',dictionary)
        entry={'run':r['run'],'subject':str(windows.subject.iloc[0]),'feature_file':str((dest/'features.pkl').resolve()),
            'windows_file':str((dest/'windows.csv').resolve()),'events_file':str((directory/'events.tsv').resolve()),
            'preprocessing_report':str((directory/'report.json').resolve()),'config':config,'rows':count,
            'features':len(feature_frame.columns),'eligible':int((y>=0).sum()),'positive':int((y==1).sum()),
            'missing_fraction':float(feature_frame.isna().to_numpy().mean()),'feature_key':key,
            'reason_counts':windows.reason.value_counts().to_dict()}
        save_json(marker,entry)
        for arr in arrays.values():
            arr._mmap.close()
        if entry['eligible']:
            entries.append(entry)
        else:
            record(config, scope='recording', identifier=r['run'], reason='no_eligible_windows')
        print(f'Features ready: {r["run"]}: {entry["features"]} features, {entry["eligible"]} eligible',flush=True)
    if not entries:
        raise ValueError('No qualified recordings have eligible forecast windows; see qualification.json')
    save_json(out/'feature_manifest.json',entries)
    return entries
