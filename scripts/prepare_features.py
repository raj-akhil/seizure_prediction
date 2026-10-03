"""Feature preparation only. Never imports or invokes the training runner."""
import argparse
from src.utils.config import load_config
from src.features.build import build_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/config.yaml')
    args=parser.parse_args()
    config=load_config(args.config)
    if not config['features']['enabled'] or not config['labels']['enabled']:
        raise ValueError('Feature preparation requires enabled features and forecast labels')
    build_features(config)


if __name__=='__main__':
    main()
