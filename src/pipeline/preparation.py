"""Orchestration of preprocessing; individual signal and label logic lives elsewhere."""
from pathlib import Path
from src.data.bids_loader import BIDSLoader, require_complete_recordings
from src.preprocessing.stream_processor import StreamProcessor, save_json
from .qualification import settings, record, qualify, report_path


def prepare_recordings(config):
    dataset=config['dataset']
    runs=BIDSLoader(dataset['raw_dir']).selected_runs(dataset)
    if not runs:
        raise ValueError('No runs match configuration')
    # Lightweight callers that have not selected an output workspace cannot
    # produce an auditable qualification report. Keep their historical strict
    # missing-modality error instead of failing later on a missing output path.
    if 'interim_dir' not in dataset:
        require_complete_recordings(runs,config)
    q=settings(config)
    # Each preprocessing invocation owns a fresh, auditable qualification report.
    report_path(config).unlink(missing_ok=True)
    if not q['enabled']:
        require_complete_recordings(runs,config)
    else:
        required=set(config['sensors']['enabled']) | {'eeg','events'}
        kept=[]
        for run in runs:
            missing=sorted(mod for mod in required if not run['files'].get(mod))
            if missing:
                record(config, scope='recording', identifier=run['id'],
                       reason='missing_required_files', detail=', '.join(missing))
            else:
                kept.append(run)
        runs=kept
        if not runs:
            raise ValueError('No selected recordings have all required files; see qualification.json')
    processor=StreamProcessor(config)
    reports=[]
    destination=Path(dataset['interim_dir'])/'latest_run.json'
    for run in runs:
        try:
            report=processor.process_run(run)
        except Exception as exc:
            report={'run':run['id'],'status':'excluded' if q['enabled'] else 'failed','error':str(exc)}
            reports.append(report)
            save_json(destination,reports)
            if not q['enabled'] or not q['continue_on_recording_error']:
                raise
            if q['enabled']:
                record(config, scope='recording', identifier=run['id'], reason='processing_error', detail=str(exc))
            else: print(report,flush=True)
            continue
        reports.append(report)
        # Save progress after every run, including interrupted invocations.
        save_json(destination,reports)
        print({k:v for k,v in report.items() if k in
               ('run','status','windows','eligible','label_counts','signal_cache_reused')},flush=True)
        if report['status']=='complete': qualify(config, run['id'])
    if any(r['status']=='failed' for r in reports):
        raise RuntimeError('Some recordings failed; see latest_run.json')
    if not any(r['status']=='complete' for r in reports):
        raise RuntimeError('No recordings passed preprocessing; see qualification.json')
    return reports
