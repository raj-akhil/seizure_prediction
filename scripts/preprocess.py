"""Command-line entry point for preprocessing only."""
import argparse
from src.utils.config import load_config
from src.pipeline.preparation import prepare_recordings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/config.yaml')
    args=parser.parse_args()
    prepare_recordings(load_config(args.config))


if __name__=='__main__':
    main()
