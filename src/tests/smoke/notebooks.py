"""Execute both authored notebooks in fresh kernels; verify their saved-model handoff.

Default: tiny synthetic multisensor fixture, never a scientific performance test.
--real: four exact local SeizeIT2 recordings; sparse EEG-only engineering check.
Generated/executed notebooks stay under audit/runs/, not in authored notebook files.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import joblib
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import numpy as np
import pandas as pd

from src.tests.smoke.training import write_edf


ROOT = Path(__file__).resolve().parents[3]


def synthetic_data(raw_dir):
    subjects = [f'sub-notebook{i:02d}' for i in range(1, 5)]
    for i, subject in enumerate(subjects):
        folder = raw_dir / subject / 'ses-01' / 'eeg'
        folder.mkdir(parents=True)
        recording_id = f'{subject}_ses-01_task-szMonitoring_run-01'
        fs, duration = 256, 600
        time = np.arange(fs * duration) / fs
        positive = ((time >= 180) & (time < 240)) | ((time >= 360) & (time < 420))
        rng = np.random.default_rng(100 + i)
        signals = np.array([(20 + 35 * positive) * np.sin(2 * np.pi * (8 + ch) * time)
                            + rng.normal(0, 5, len(time)) for ch in range(2)])
        path = folder / f'{recording_id}_eeg.edf'
        write_edf(path, signals, ['BTEright SD', 'CROSStop SD'], fs, 'uV', 2000)
        path.with_suffix('.json').write_text(json.dumps({'SamplingFrequency': fs, 'RecordingDuration': duration}))
        pd.DataFrame({'onset': [300., 480.], 'duration': [5., 5.],
                      'eventType': ['sz_foc', 'sz_foc'], 'recordingDuration': [duration, duration]}).to_csv(
                          folder / f'{recording_id}_events.tsv', sep='\t', index=False)
        for mod, names, rate in [
            ('ecg', ['ECG SD'], 256),
            ('emg', ['EMG SD'], 256),
            ('mov', ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z'], 25),
        ]:
            sensor_folder = folder.parent / mod
            sensor_folder.mkdir()
            sensor_time = np.arange(rate * duration) / rate
            if mod == 'ecg':
                values = (800 * np.exp(-((sensor_time % 1 - 0.5) / 0.02) ** 2)
                          + rng.normal(0, 5, len(sensor_time)))[None, :]
            else:
                values = rng.normal(0, 0.1 if mod == 'mov' else 20, (len(names), len(sensor_time)))
            sensor_file = sensor_folder / f'{recording_id}_{mod}.edf'
            write_edf(sensor_file, values, names, rate, 'g' if mod == 'mov' else 'uV', 4 if mod == 'mov' else 2000)
            sensor_file.with_suffix('.json').write_text(json.dumps({'SamplingFrequency': rate, 'RecordingDuration': duration}))
    return subjects


def execute_notebook(name, overrides, target):
    notebook = nbformat.read(ROOT / name, as_version=4)
    nbformat.validate(notebook)
    parameter_cells = [cell for cell in notebook.cells if 'parameters' in cell.metadata.get('tags', [])]
    if len(parameter_cells) != 1:
        raise ValueError(f'{name} must have one parameters cell')
    # Override the published parameter cell without modifying the delivered notebook.
    parameter_cells[0].source += '\n# Engineering-check overrides\n' + '\n'.join(
        f'{key} = {value!r}' for key, value in overrides.items())
    kernel = KernelManager(kernel_name='python3')
    kernel.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=kernel, timeout=300,
                            resources={'metadata': {'path': str(ROOT)}})

    def progress(cell, cell_index, **_):
        if cell.cell_type == 'code':
            print(f'{name}: cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

    client.on_cell_start = progress
    try:
        client.execute()
    finally:
        nbformat.write(notebook, target / name)
        # An explicitly supplied kernel manager is caller-owned.
        if kernel.has_kernel:
            kernel.shutdown_kernel(now=True)
        kernel.cleanup_resources()
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name = f'smoke_{"real" if args.real else "synthetic"}_{stamp}'
    target = ROOT / 'audit' / 'runs' / 'smoke_tests' / 'notebooks' / stamp
    target.mkdir(parents=True)
    if args.real:
        subjects = ['sub-001', 'sub-002', 'sub-004', 'sub-007']
        recordings = [f'{subject}_ses-01_task-szMonitoring_run-{run}' for subject, run in
                      zip(subjects, ['03', '01', '01', '03'])]
        raw_dir = ROOT / 'data' / 'raw'
        preparation = {'STRIDE_SEC': 120, 'SUBJECTS': subjects, 'RECORDINGS': recordings}
    else:
        raw_dir = target / 'synthetic_raw'
        subjects = synthetic_data(raw_dir)
        preparation = {'STRIDE_SEC': 5, 'MINIMUM_WARNING_MIN': 1, 'OCCURRENCE_PERIOD_MIN': 1,
                       'POSTICTAL_EXCLUSION_MIN': 0.1}
    preparation.update(DATASET_NAME=name, RAW_DIR=str(raw_dir))
    if args.real:
        preparation['MODALITIES'] = ['eeg']
    training = {'DATASET_NAME': name, 'INNER_FOLDS': 2, 'FEATURE_K': 10,
                'MIN_EVENTS_PER_PARTITION': 2,
                'PATIENT_SPLITS': {'train': subjects[:2], 'validation': subjects[2:3], 'test': subjects[3:]},
                'MODEL_GRIDS': {'logistic': {'C': [1.0], 'max_iter': [1000], 'class_weight': ['balanced']}}}
    prepared = execute_notebook('01_prepare_data.ipynb', preparation, target)
    trained = execute_notebook('02_train_model.ipynb', training, target)
    workspace = ROOT / 'audit' / 'runs' / 'notebooks' / name
    handoff = json.loads((workspace / 'handoff.json').read_text())
    completed = list((workspace / 'training').glob('*/complete.json'))
    if len(completed) != 1:
        raise AssertionError('Expected exactly one completed training experiment')
    run_dir = completed[0].parent
    artifact = joblib.load(run_dir / 'model.joblib')
    predictions = pd.read_csv(run_dir / 'test_predictions.csv')
    entries = json.loads((run_dir / 'feature_manifest.json').read_text())
    features = []
    for entry in entries:
        if entry['subject'] not in training['PATIENT_SPLITS']['test']:
            continue
        windows = pd.read_csv(entry['windows_file'])
        features.append(pd.read_pickle(entry['feature_file']).loc[windows.eligible])
    test_x = pd.concat(features, ignore_index=True, sort=False)
    actual = artifact['model'].predict_proba(test_x)[:, 1]
    np.testing.assert_allclose(actual, predictions.probability.to_numpy(), rtol=1e-12, atol=1e-12)
    # The existing reporting CLI must accept the notebook's experiment contract.
    subprocess.run([sys.executable, '-m', 'src.commands.validate_model', '--run-dir', str(run_dir)],
                   cwd=ROOT, check=True)
    if not any('image/png' in output.get('data', {}) for notebook in [prepared, trained]
               for cell in notebook.cells for output in cell.get('outputs', [])):
        raise AssertionError('Expected rendered notebook diagnostic figures')
    summary = {'status': 'passed', 'mode': 'real EEG-only sparse check' if args.real else 'synthetic check',
               'prepared_eligible_windows': handoff['eligible_windows'],
               'positive_windows': handoff['positive_windows'], 'model_directory': str(run_dir),
               'executed_notebooks': str(target), 'saved_model_reload_predictions_match': True,
               'report_regeneration': 'passed', 'purpose': 'execution verification; not scientific validation'}
    (target / 'RESULTS.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
