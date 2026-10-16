"""Audit every saved window from the latest preprocessing invocation."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.prepared import PreparedRun

from src.utils.config import load_config
import argparse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='src/config/config.yaml')
    parser.add_argument('--output-dir',default='audit/runs/reports/preprocessing')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    report_dir=Path(args.output_dir)
    if not report_dir.is_absolute():
        report_dir=root/report_dir
    report_dir.mkdir(parents=True,exist_ok=True)
    c=load_config(args.config)
    out=Path(c['dataset']['interim_dir'])
    reports = json.loads((out/'latest_run.json').read_text())
    labels, reasons, schemas = Counter(), Counter(), Counter()
    rows, errors = [], []
    hours = 0
    size = 0
    for r in reports:
        if r['status'] != 'complete':
            errors.append(r)
            continue
        directory = Path(r['result_directory'])
        cfg = json.loads((directory/'config.json').read_text())
        windows = pd.read_csv(directory/'windows.csv')
        y = np.load(directory/'y.npy')
        quality = np.load(Path(r['signal_directory'])/'quality_valid.npy')
        assert len(windows) == len(y) == r['windows'] == len(quality), r['run']
        np.testing.assert_array_equal(y,windows.label)
        np.testing.assert_array_equal(quality,windows.signal_valid)
        np.testing.assert_array_equal(y>=0,windows.eligible)
        assert set(y).issubset({-1,0,1})
        assert not np.any((y>=0)&~quality)
        duration = r['inventory']['eeg']['duration_sec']
        hours += duration/3600
        horizon = 60*(cfg['labels']['minimum_warning_min']+cfg['labels']['occurrence_period_min'])
        assert (windows.loc[y>=0,'prediction_time_sec']+horizon<=duration+1e-6).all()
        assert (windows.processing_support_start_sec>=0).all()
        assert (windows.prediction_time_sec<=duration).all()
        assert np.allclose(np.diff(windows.prediction_time_sec),cfg['preprocessing']['stride_sec'])
        events = pd.read_csv(directory/'events.tsv',sep='\t')
        for _, w in windows.loc[y==1].iterrows():
            lo=w.prediction_time_sec+cfg['labels']['minimum_warning_min']*60
            hi=lo+cfg['labels']['occurrence_period_min']*60
            assert ((events.onset>=lo)&(events.onset<hi)&events.eventType.str.startswith(tuple(cfg['labels']['seizure_prefixes']))).any()
        names=[]
        for mod, info in r['inventory'].items():
            path=Path(r['signal_directory'])/f'{mod}.npy'
            size+=path.stat().st_size
            a=np.load(path,mmap_mode='r')
            assert list(a.shape)==info['shape']
            assert a.shape[0]==len(y) and a.shape[1]==len(info['channels']) and a.dtype==np.float32
            assert a.shape[2]==round((cfg['preprocessing']['window_sec']+cfg['preprocessing']['context_sec'])*info['output_rate'])
            for i in range(0,len(y),128):
                assert np.isfinite(a[i:i+128][quality[i:i+128]]).all(), (r['run'],mod)
            del a
            names.extend(f'{mod}:{ch}' for ch in info['channels'])
        schema=' | '.join(names)
        schemas[schema]+=1
        labels.update(map(int,y))
        reasons.update(windows.reason)
        if (y>=0).any():
            d=PreparedRun(directory)
            common_rates={info['output_rate'] for info in r['inventory'].values()}
            x,meta=d.get(0,combined=len(common_rates)==1)
            if len(common_rates)==1:
                expected_samples=round((cfg['preprocessing']['window_sec']+cfg['preprocessing']['context_sec'])*next(iter(common_rates)))
                assert x.shape==(len(names),expected_samples)
            assert meta['eligible']
            del x
            d.close()
        rows.append({'run':r['run'],'windows':len(y),'eligible':int((y>=0).sum()),'positive':int((y==1).sum()),'channels':len(names)})
    summary={'runs':len(reports),'completed':len(rows),'errors':errors,'recorded_hours':hours,
        'windows':sum(labels.values()),'label_counts':dict(labels),'reason_counts':dict(reasons),
        'array_storage_GiB':size/1024**3,'channel_schemas':dict(schemas),'zero_eligible_runs':[r['run'] for r in rows if r['eligible']==0],
        'positive_runs':[r for r in rows if r['positive']], 'checks':'All saved windows: shapes, dtype, finite valid signals, label consistency, timing boundaries, positive-event matching; reader checked per eligible run',
        'alignment_limitation':'Publisher run-relative alignment assumed, not independently signal-verified'}
    (report_dir/'full_validation.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(rows).to_csv(report_dir/'full_validation_runs.csv',index=False)
    print(json.dumps(summary,indent=2))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
