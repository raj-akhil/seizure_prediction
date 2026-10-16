"""Metadata-only cohort checks; no waveform processing or model fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import yaml

from src.dashboard.patients import patient_catalog, suggest_split


class PatientSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name)
        self.config = yaml.safe_load(Path('src/config/real_data_check.yaml').read_text())
        self.config['dataset'].update(subjects=None, runs=None, recordings=None)
        self.config['training']['splits'] = {'train': [], 'validation': [], 'test': []}

    def recording(self, subject, onset=951, missing=()):
        stem = f'{subject}_ses-01_task-szMonitoring_run-01'
        for mod in ('eeg', 'ecg', 'emg', 'mov'):
            folder = self.raw / subject / 'ses-01' / mod
            folder.mkdir(parents=True, exist_ok=True)
            if mod not in missing:
                # Deliberately not an EDF: preview must never read signal samples.
                (folder / f'{stem}_{mod}.edf').write_bytes(b'not waveform data')
                (folder / f'{stem}_{mod}.json').write_text(json.dumps({'RecordingDuration': 4000}))
        (self.raw / subject / 'ses-01/eeg' / f'{stem}_events.tsv').write_text(
            f'onset\tduration\teventType\n{onset}\t30\tsz_foc_f2b\n')
        return stem

    def test_refresh_discovers_new_folders_and_tracks_missing_modalities(self):
        self.recording('sub-1', missing=('mov',))
        initial = patient_catalog(self.raw, self.config)
        self.assertEqual(initial['patients'][0]['incomplete_recordings'], 1)
        self.assertIsNone(initial['patients'][0]['potential_events'])
        self.recording('sub-2')
        self.assertEqual(len(patient_catalog(self.raw, self.config)['patients']), 2)
        self.config['sensors']['enabled'] = ['eeg', 'ecg']
        self.assertEqual(patient_catalog(self.raw, self.config)['patients'][0]['incomplete_recordings'], 0)

    def test_forecast_horizon_and_seizure_selection_change_event_preview(self):
        self.recording('sub-1')
        self.assertEqual(patient_catalog(self.raw, self.config)['patients'][0]['potential_events'], 1)
        self.config['labels']['minimum_warning_min'] = 30
        self.assertEqual(patient_catalog(self.raw, self.config)['patients'][0]['potential_events'], 0)
        self.config['labels']['minimum_warning_min'] = 5
        self.config['labels']['seizure_prefixes'] = ['sz_gen']
        self.assertEqual(patient_catalog(self.raw, self.config)['patients'][0]['potential_events'], 0)

    def test_fast_listing_defers_sidecar_and_event_inspection(self):
        self.recording('sub-1')
        next(self.raw.rglob('*_eeg.json')).unlink()
        catalog = patient_catalog(self.raw, self.config, include_timing=False)
        self.assertFalse(catalog['timing_checked'])
        self.assertEqual(catalog['patients'][0]['recordings'], 1)
        self.assertIsNone(catalog['patients'][0]['potential_events'])
        self.assertIn('Fast listing', catalog['note'])

    def test_recording_filters_and_missing_configured_patients_are_visible(self):
        first = self.recording('sub-1')
        self.recording('sub-2')
        self.config['dataset'].update(subjects=['sub-2', 'sub-missing'], recordings=[first])
        catalog = patient_catalog(self.raw, self.config)
        rows = {row['id']: row for row in catalog['patients']}
        self.assertFalse(rows['sub-1']['selected'])
        self.assertEqual(rows['sub-2']['recordings'], 0)
        self.assertEqual(rows['sub-missing']['recordings'], 0)
        self.assertTrue(catalog['warnings'])
        self.assertIsNone(suggest_split(catalog, self.config)['splits'])

    def test_suggestion_is_reproducible_disjoint_and_does_not_modify_config(self):
        for i in range(4):
            self.recording(f'sub-{i}')
        original = copy.deepcopy(self.config)
        catalog = patient_catalog(self.raw, self.config)
        suggestion = suggest_split(catalog, self.config)
        self.assertEqual(suggestion, suggest_split(catalog, self.config))
        groups = suggestion['splits']
        self.assertEqual([len(groups[key]) for key in ('train', 'validation', 'test')], [2, 1, 1])
        assigned = sum(groups.values(), [])
        self.assertEqual(len(set(assigned)), 4)
        self.assertEqual(self.config, original)
        self.config['training']['inner_folds'] = 3
        self.assertIsNone(suggest_split(catalog, self.config)['splits'])

    def test_insufficient_events_or_unknown_metadata_do_not_suggest_a_ready_split(self):
        for i in range(4):
            self.recording(f'sub-{i}', onset=100 if i == 0 else 951)
        result = suggest_split(patient_catalog(self.raw, self.config), self.config)
        self.assertIsNone(result['splits'])
        self.assertTrue(any('Only 3 patients' in text for text in result['reasons']))
        self.recording('sub-0')
        next((self.raw / 'sub-0').rglob('*_eeg.json')).unlink()
        catalog = patient_catalog(self.raw, self.config)
        self.assertIsNone(catalog['patients'][0]['potential_events'])
        self.assertIsNone(suggest_split(catalog, self.config)['splits'])


if __name__ == '__main__':
    unittest.main()
