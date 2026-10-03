"""Report saved held-out predictions without fitting or changing the selected threshold."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from src.validation.report import write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,help='Completed trusted local training run')
    args=parser.parse_args()
    root=Path(args.run_dir).resolve()
    if not (root/'complete.json').exists():
        raise ValueError('Training run is incomplete')
    config=json.loads((root/'config.json').read_text())
    evaluation=json.loads((root/'evaluation.json').read_text())
    entries=json.loads((root/'feature_manifest.json').read_text())
    events={e['run']:pd.read_csv(e['events_file'],sep='\t') for e in entries}
    reference=root/'validation_reference.json'
    if reference.exists():
        prevalence=json.loads(reference.read_text())['training_prevalence']
    else:
        # Compatibility with older runs: use saved training metadata, never held-out labels.
        labels=[]
        for entry in entries:
            if entry['subject'] in config['training']['splits']['train']:
                windows=pd.read_csv(entry['windows_file'])
                labels.extend(windows.loc[windows.eligible.astype(bool),'label'].tolist())
        if not labels:
            raise ValueError('Training-only reference unavailable')
        prevalence=sum(labels)/len(labels)
    splits={key:set(value) for key,value in config['training']['splits'].items()}
    if any(splits[a]&splits[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Overlapping patient partitions')
    target=root/'validation_reports'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for partition in ['validation','test']:
        frame=pd.read_csv(root/f'{partition}_predictions.csv')
        if set(frame.subject)!=splits[partition]:
            raise ValueError('Prediction patients do not match saved partitions')
        write_report(frame,config,events,evaluation['threshold'],prevalence,target/partition,partition)
    print(f'Validation reports: {target}')


if __name__=='__main__':
    main()
