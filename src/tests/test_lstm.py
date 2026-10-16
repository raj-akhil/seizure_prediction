import copy
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from src.training.lstm import build_sequences, fit
from src.training.runner import train
from src.utils.config import load_config


class LSTMSequenceTests(unittest.TestCase):
    def test_sequences_respect_recording_and_gap_boundaries(self):
        values = np.arange(14, dtype=float).reshape(7, 2)
        metadata = pd.DataFrame({
            'recording_id': ['one'] * 4 + ['two'] * 3,
            'prediction_time_sec': [0, 5, 10, 25, 0, 5, 10],
        })
        sequences, endpoints = build_sequences(values, metadata, sequence_windows=2, stride_sec=5)
        # [0,5] and [5,10] for each recording; the 15-second gap cannot bridge.
        np.testing.assert_array_equal(endpoints, [1, 2, 5, 6])
        self.assertEqual(sequences.shape, (4, 2, 2))
        np.testing.assert_array_equal(sequences[0], values[[0, 1]])
        np.testing.assert_array_equal(sequences[-1], values[[5, 6]])

    def test_lstm_inferrs_selected_feature_width_and_reloads(self):
        config = copy.deepcopy(load_config())
        config['training']['lstm'].update(enabled=True, sequence_windows=3, feature_k='all',
                                          hidden_size=4, batch_size=8, max_epochs=1,
                                          patience=1, max_training_sequences=24)
        rows = []
        for subject_index, subject in enumerate(['a', 'b', 'c']):
            for window in range(8):
                rows.append({'subject': subject, 'recording_id': subject,
                             'prediction_time_sec': window * 5, 'label': window % 2,
                             'event_row': window // 2})
        metadata = pd.DataFrame(rows)
        feature = np.arange(len(rows), dtype=float)
        X = pd.DataFrame({'eeg::C3::rms': feature,
                          'ecg::ECG::mean': (feature % 5) + metadata.label.to_numpy()})
        train_rows = np.arange(len(rows))
        folds = []
        for held_out in range(3):
            validation = np.arange(held_out * 8, (held_out + 1) * 8)
            training = np.setdiff1d(train_rows, validation)
            folds.append((training, validation))
        _, model, details = fit(config, X, metadata, train_rows, folds)
        self.assertEqual(details['input_features'], 2)
        kept_rows, probabilities = model.predict_for_rows(X, metadata, train_rows)
        self.assertEqual(len(kept_rows), 18)
        self.assertTrue(np.isfinite(probabilities).all())
        with tempfile.TemporaryDirectory() as directory:
            path = f'{directory}/lstm.joblib'
            joblib.dump(model, path)
            restored = joblib.load(path)
            restored_rows, restored_probabilities = restored.predict_for_rows(X, metadata, train_rows)
        np.testing.assert_array_equal(restored_rows, kept_rows)
        np.testing.assert_allclose(restored_probabilities, probabilities)

    def test_runner_writes_sequence_only_evaluation_artifacts(self):
        config = copy.deepcopy(load_config())
        config['labels'].update(minimum_warning_min=0, occurrence_period_min=1)
        config['training'].update(output_dir='', min_positive_events_per_partition=1,
                                  splits={'train': ['a', 'b', 'c'], 'validation': ['d'], 'test': ['e']})
        config['training']['lstm'].update(enabled=True, sequence_windows=3, feature_k='all',
                                          hidden_size=4, batch_size=8, max_epochs=1,
                                          patience=1, max_training_sequences=24)
        config['validation'].update(bootstrap_replicates=0, plots=False)
        records = []
        for subject_index, subject in enumerate(['a', 'b', 'c', 'd', 'e']):
            for window in range(14):
                time = window * 5
                records.append({'subject': subject, 'recording_id': subject,
                                'prediction_time_sec': time, 'label': int(time <= 50),
                                'event_row': 0, 'eligible': True})
        metadata = pd.DataFrame(records)
        sequence = np.arange(len(records), dtype=float)
        X = pd.DataFrame({'eeg::C3::rms': sequence,
                          'ecg::ECG::mean': (sequence % 7) + metadata.label.to_numpy()})
        indices = {'train': np.arange(42), 'validation': np.arange(42, 56), 'test': np.arange(56, 70)}
        folds = []
        for held_out in range(3):
            validation = np.arange(held_out * 14, (held_out + 1) * 14)
            folds.append((np.setdiff1d(np.arange(42), validation), validation))
        events = {subject: pd.DataFrame({'onset': [50.], 'duration': [5.], 'eventType': ['sz_foc']})
                  for subject in ['a', 'b', 'c', 'd', 'e']}
        with tempfile.TemporaryDirectory() as directory:
            config['training']['output_dir'] = directory
            output = train(config, X, metadata, events, [], indices, folds)
            evaluation = (output / 'evaluation.json').read_text()
            self.assertIn('feature_lstm', evaluation)
            saved = pd.read_csv(output / 'test_predictions.csv')
            # First two windows in each recording are omitted because their full
            # trailing sequence is not available; no padded/future rows are used.
            self.assertEqual(len(saved), 12)
            self.assertTrue((output / 'validation_reports' / 'test' / 'metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
