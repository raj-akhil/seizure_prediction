"""Validate extracted feature artifacts without fitting a model or selector."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from src.utils.config import load_config

import argparse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='src/config/config.yaml')
    parser.add_argument('--output-dir',default='audit/runs/reports/features')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    report_dir=Path(args.output_dir)
    if not report_dir.is_absolute():
        report_dir=root/report_dir
    report_dir.mkdir(parents=True,exist_ok=True)
    c=load_config(args.config)
    out=Path(c['dataset']['interim_dir'])
    entries=json.loads((out/'feature_manifest.json').read_text())
    labels=Counter();reasons=Counter();widths=Counter();names=set();total=0;bytes_used=0
    cardiac=[]
    for e in entries:
        x=pd.read_pickle(e['feature_file'])
        w=pd.read_csv(e['windows_file'])
        assert len(x)==len(w)==e['rows']
        assert x.columns.is_unique and len(x.columns)==e['features']
        assert all(col.startswith(('eeg::','ecg::','emg::','mov::')) for col in x)
        assert not np.isinf(x.to_numpy()).any()
        assert np.array_equal(w.eligible,w.label>=0)
        assert set(w.label).issubset({-1,0,1})
        assert (w.loc[w.eligible,'feature_support_start_sec']>=0).all()
        assert (w.feature_support_start_sec<=w.input_start_sec).all()
        if 'ecg' in c['sensors']['enabled'] and 'cardiac' in c['features']['families']['ecg']:
            assert (w.loc[w.eligible,'prediction_time_sec']-w.loc[w.eligible,'feature_support_start_sec']>=c['features']['ecg']['history_sec']-1e-6).all()
        events=pd.read_csv(e['events_file'],sep='\t')
        duration=float(events.recordingDuration.iloc[0])
        for _,row in w.loc[w.label==1].iterrows():
            lo=row.prediction_time_sec+c['labels']['minimum_warning_min']*60
            hi=lo+c['labels']['occurrence_period_min']*60
            assert hi<=duration+1e-6
            assert ((events.onset>=lo)&(events.onset<hi)&events.eventType.str.startswith(tuple(c['labels']['seizure_prefixes']))).any()
        labels.update(map(int,w.label));reasons.update(w.reason);widths[len(x.columns)]+=1
        total+=len(x);names.update(x.columns);bytes_used+=Path(e['feature_file']).stat().st_size
        cardiac.extend(x.filter(regex='heart_rate_bpm$').to_numpy().ravel().tolist())
    summary={'runs':len(entries),'windows':total,'unique_feature_columns':len(names),'feature_widths':dict(widths),
        'label_counts':dict(labels),'reason_counts':dict(reasons),'feature_storage_MiB':bytes_used/1024**2,
        'estimated_hr_median_bpm':float(np.nanmedian(cardiac)) if np.isfinite(cardiac).any() else None,
        'validation':'Row alignment, named numeric features, no infinities, eligibility, historical support and positive-event matching passed',
        'training_started':False,'fitted_feature_selection_started':False,
        'limitations':['Publisher timing assumption remains','RR detector not validated against annotated beats',
            'This artifact audit does not estimate predictive performance']}
    (report_dir/'FEATURE_RESULTS.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
