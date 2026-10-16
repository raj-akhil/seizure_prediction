"""Read preprocessed windows without loading entire recordings into RAM."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


class PreparedRun:
    def __init__(self, result_directory, eligible_only=True):
        path = Path(result_directory)
        self.report = json.loads((path/'report.json').read_text())
        self.windows = pd.read_csv(path/'windows.csv')
        self.indices = np.flatnonzero(self.windows.eligible) if eligible_only else np.arange(len(self.windows))
        self.arrays = {mod:np.load(Path(self.report['signal_directory'])/f'{mod}.npy',mmap_mode='r')
                       for mod in self.report['inventory']}
        self.channel_names = [f'{mod}:{ch}' for mod,info in self.report['inventory'].items() for ch in info['channels']]

    def __len__(self):
        return len(self.indices)

    def get(self, index, combined=False):
        i = int(self.indices[index])
        signals = {mod:array[i] for mod,array in self.arrays.items()}
        if combined:
            rates = {info['output_rate'] for info in self.report['inventory'].values()}
            if len(rates) != 1:
                raise ValueError('Native-rate modalities require separate inputs; configure a common output rate to concatenate')
            signals = np.concatenate(list(signals.values()),axis=0)
        return signals, self.windows.iloc[i].to_dict()

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()
