"""Engineering tests only: synthetic multi-subject EDF pipeline and real-data fitting."""
import copy
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from sklearn.model_selection import ParameterGrid
from src.utils.config import load_config
from src.training.data import load_feature_data
from src.training.splits import preflight
from src.training.registry import candidates
from src.training.runner import train
from src.preprocessing.stream_processor import save_json


def write_edf(path, signals, names, fs, unit, limit):
    """Small deterministic EDF fixture writer; one-second records, signed 16-bit."""
    n=len(names);seconds=signals.shape[1]//fs
    def field(value,width):
        text=str(value)
        if len(text)>width:
            raise ValueError('EDF field too long')
        return text.ljust(width).encode('ascii')
    header=b''.join(field(v,w) for v,w in [
        ('0',8),('SYNTHETIC',80),('ENGINEERING TEST ONLY',80),('01.01.00',8),('00.00.00',8),
        (256+256*n,8),('',44),(seconds,8),(1,8),(n,4)])
    for values,width in [(names,16),(['']*n,80),([unit]*n,8),([-limit]*n,8),([limit]*n,8),
                         ([-32768]*n,8),([32767]*n,8),(['']*n,80),([fs]*n,8),(['']*n,32)]:
        header+=b''.join(field(v,width) for v in values)
    digital=np.rint((np.clip(signals,-limit,limit)+limit)/(2*limit)*65535-32768).astype('<i2')
    with path.open('wb') as f:
        f.write(header)
        for second in range(seconds):
            f.write(digital[:,second*fs:(second+1)*fs].tobytes())


def small_config(base, output):
    c=copy.deepcopy(base)
    c['training']['output_dir']=str(output)
    c['training']['selection']['k']=10
    c['training']['selection_grid']={'k':[8,16]}
    c['training']['models']={
        'logistic':{'C':[1.0],'max_iter':[1000],'class_weight':['balanced']},
        'random_forest':{'n_estimators':[12],'max_depth':[4],'min_samples_leaf':[2],'class_weight':['balanced']},
        'hist_gradient_boosting':{'max_iter':[12],'max_leaf_nodes':[7],'min_samples_leaf':[5],'class_weight':['balanced']}}
    return c


def main():
    root=Path(__file__).resolve().parents[3]
    target=root / 'audit' / 'runs'/'smoke_tests'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True,exist_ok=False)
    base=load_config()
    c=small_config(base,target/'synthetic_training')
    c['dataset'].update(raw_dir=str(target/'synthetic_raw'),interim_dir=str(target/'synthetic_interim'),subjects=None,runs=None)
    c['labels'].update(minimum_warning_min=1,occurrence_period_min=1,postictal_exclusion_min=.1)
    subjects=[f'sub-synthetic{i}' for i in range(5)]
    c['training']['splits']={'train':subjects[:3],'validation':subjects[3:4],'test':subjects[4:]}
    c['training']['alarms'].update(consecutive_windows=1,refractory_sec=120,max_false_alarms_per_day=1000000)
    # Smoke budget is deliberately permissive to test serialization/evaluation even for weak candidates.
    c['alignment']['evidence']='Synthetic fixtures share a known common origin; not clinical data'
    for k,subject in enumerate(subjects):
        rng=np.random.default_rng(100+k)
        run_id=f'{subject}_ses-01_task-szMonitoring_run-01'
        for mod,names,fs in [('eeg',['BTEright SD','CROSStop SD'],256),('ecg',['ECG SD'],256),
                              ('emg',['EMG SD'],256),('mov',['EEG SD ACC X','EEG SD ACC Y','EEG SD ACC Z'],25)]:
            folder=target/'synthetic_raw'/subject/'ses-01'/mod
            folder.mkdir(parents=True,exist_ok=True)
            time=np.arange(600*fs)/fs
            pre=((time>=180)&(time<240))|((time>=360)&(time<420))
            if mod=='ecg':
                data=(800*np.exp(-((time%1-.5)/.02)**2)+rng.normal(0,5,len(time)))[None,:]
            elif mod=='eeg':
                data=np.array([(20+40*pre)*np.sin(2*np.pi*(8+j)*time)+rng.normal(0,5,len(time)) for j in range(len(names))])
            elif mod=='emg':
                data=rng.normal(0,20,(1,len(time)))
            else:
                data=rng.normal(0,.1,(len(names),len(time)))
            path=folder/(run_id+f'_{mod}.edf')
            write_edf(path,data,names,fs,'g' if mod=='mov' else 'uV',4 if mod=='mov' else 2000)
            save_json(path.with_suffix('.json'),{'SamplingFrequency':fs,'RecordingDuration':600})
            if mod=='eeg':
                pd.DataFrame({'onset':[300.,480.],'duration':[5.,5.],'eventType':['sz_foc','sz_foc'],
                    'recordingDuration':[600.,600.]}).to_csv(folder/(run_id+'_events.tsv'),sep='\t',index=False)
    cfg=target/'synthetic_config.yaml'
    cfg.write_text(yaml.safe_dump(c,sort_keys=False))
    for module in ('src.commands.preprocess','src.commands.prepare_features'):
        subprocess.run([sys.executable,'-m',module,'--config',str(cfg)],check=True,cwd=root)
    X,meta,events,entries=load_feature_data(c)
    indices,folds=preflight(c,X,meta)
    model_dir=train(c,X,meta,events,entries,indices,folds)
    artifact=joblib.load(model_dir/'model.joblib')
    actual=artifact['model'].predict_proba(X.iloc[indices['test']])[:,1]
    saved=pd.read_csv(model_dir/'test_predictions.csv').probability.to_numpy()
    np.testing.assert_allclose(actual,saved,rtol=1e-12,atol=1e-12)
    # Real data is optional. The synthetic smoke test must remain runnable after a
    # clean dashboard workspace, where audit/runs/cli_prepared has intentionally been removed.
    real_results=[]
    try:
        rx,rm,_,_=load_feature_data(base)
    except FileNotFoundError:
        real_results.append({'status':'skipped','reason':'No local prepared real-data feature manifest'})
    else:
        # Explicitly an in-sample fit compatibility check, with no purported validation split.
        positive=np.flatnonzero(rm.label.eq(1))[:60]
        negative=np.flatnonzero(rm.label.eq(0))[::max(1,int(rm.label.eq(0).sum()/180))][:180]
        ids=np.concatenate([positive,negative])
        if not len(positive) or not len(negative):
            raise ValueError('Real smoke subset needs both classes')
        for name,(pipeline,grid) in candidates(small_config(base,target)).items():
            pipeline.set_params(**next(iter(ParameterGrid(grid))))
            pipeline.fit(rx.iloc[ids],rm.label.iloc[ids])
            probabilities=pipeline.predict_proba(rx.iloc[ids])[:,1]
            assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
            real_results.append({'model':name,'rows':len(ids),'selected_features':len(pipeline['features'].get_feature_names_out()),
                'fit_and_predict':'passed','evaluation':'No generalization score; in-sample engineering check only'})
    result={'status':'passed','purpose':'Engineering smoke test, not clinical or population validation',
        'synthetic_subjects':5,'synthetic_eligible_windows':len(meta),'grouped_inner_folds':len(folds),
        'synthetic_partitions':{k:len(v) for k,v in indices.items()},'synthetic_model_directory':str(model_dir),
        'saved_model_reload_predictions_match':True,'real_data_smoke':real_results,
        'full_real_population_training_started':False}
    save_json(target/'SMOKE_RESULTS.json',result)
    save_json(root / 'audit' / 'runs'/'smoke_tests'/'latest.json',{'report':str(target/'SMOKE_RESULTS.json')})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
