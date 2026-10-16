"""Default is a no-fit readiness check. Training requires --start-training."""
import argparse
import json
from pathlib import Path
from src.utils.config import load_config
from src.models.training import load_feature_data, preflight, candidates, train
from src.training.lstm import enabled as lstm_enabled
from src.training.readiness import report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='src/config/config.yaml')
    parser.add_argument('--start-training',action='store_true',help='Explicitly start fitting and final held-out evaluation')
    args=parser.parse_args()
    config=load_config(args.config)
    X,meta,events,entries=load_feature_data(config)
    readiness=report(config,meta)
    destination=Path(config['training']['output_dir'])
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'readiness.json').write_text(json.dumps(readiness,indent=2),encoding='utf-8')
    active_models=['feature_lstm'] if lstm_enabled(config) else list(candidates(config))
    print(json.dumps({'mode':'TRAIN' if args.start_training else 'CHECK_ONLY_NO_FIT','subjects':sorted(meta.subject.unique()),
        'eligible_windows':len(meta),'feature_columns':len(X.columns),'positive_windows':int(meta.label.sum()),
        'candidate_models':active_models},indent=2))
    print('PARTITION READINESS: '+json.dumps(readiness['partitions'],indent=2))
    try:
        indices,folds=preflight(config,X,meta)
    except ValueError as exc:
        print(f'NOT READY: {exc}')
        raise SystemExit(2)
    if not args.start_training:
        print('READY. No model or feature selector was fitted.')
        return
    print(f'Training results: {train(config,X,meta,events,entries,indices,folds)}')


if __name__=='__main__':
    main()
