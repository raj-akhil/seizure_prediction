import unittest
from unittest.mock import patch
from src.data.bids_loader import BIDSLoader, require_complete_recordings
from src.pipeline.preparation import prepare_recordings


class RecordingSelectionTests(unittest.TestCase):
    def fixture(self):
        return [{'id':f'{s}_ses-01_task-szMonitoring_{r}','subject':s,'run':r,
                 'files':{'eeg':'signal.edf','events':'events.tsv'}}
                for s in ['sub-1','sub-2'] for r in ['run-01','run-02']]

    def test_exact_recordings_do_not_select_cross_patient_run_numbers(self):
        rows=self.fixture()
        with patch.object(BIDSLoader,'discover_runs',return_value=rows):
            chosen=BIDSLoader('.').selected_runs({'recordings':[rows[0]['id'],rows[3]['id']]})
        self.assertEqual(chosen,[rows[0],rows[3]])

    def test_unknown_or_conflicting_filters_refused(self):
        rows=self.fixture()
        with patch.object(BIDSLoader,'discover_runs',return_value=rows):
            with self.assertRaisesRegex(ValueError,'missing or excluded'):
                BIDSLoader('.').selected_runs({'subjects':['sub-2'],'recordings':[rows[0]['id']]})

    def test_missing_modality_explains_recording_before_signal_processing(self):
        config={'dataset':{'raw_dir':'.'},'sensors':{'enabled':['eeg','ecg'],'missing_policy':'error'}}
        with patch.object(BIDSLoader,'discover_runs',return_value=self.fixture()),patch('src.pipeline.preparation.StreamProcessor') as processor:
            with self.assertRaisesRegex(ValueError,'sub-1.*ecg'):
                prepare_recordings(config)
            processor.assert_not_called()
