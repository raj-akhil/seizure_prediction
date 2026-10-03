import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.data.bids_loader import BIDSLoader
from src.preprocessing.labels import label_windows, read_events
from src.preprocessing.stream_processor import StreamProcessor, integer_samples


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()['labels']
        self.cfg.update(minimum_warning_min=1, occurrence_period_min=1, postictal_exclusion_min=1)
        self.events = pd.DataFrame([{'onset':180., 'duration':10., 'eventType':'sz_foc'}])

    def labels(self, ends, events=None, duration=600):
        ends = np.array(ends,dtype=float)
        return label_windows(ends-5,ends,self.events if events is None else events,duration,self.cfg)

    def test_half_open_forecast_boundaries(self):
        y,_,_ = self.labels([60,65,120,125,180,185,250,255])
        np.testing.assert_array_equal(y,[0,1,1,-1,-1,-1,-1,0])

    def test_unobserved_future_is_not_background(self):
        y,reasons,_ = self.labels([480,485])
        self.assertEqual(y.tolist(),[0,-1])
        self.assertEqual(reasons[1],'unobserved_future')

    def test_unselected_seizure_excluded(self):
        self.cfg['seizure_prefixes'] = ['sz_gen']
        y,reasons,_ = self.labels([100])
        self.assertEqual(y[0],-1)
        self.assertEqual(reasons[0],'unselected_seizure')

    def test_cluster_overrides_positive(self):
        events = pd.concat([self.events,pd.DataFrame([{'onset':110.,'duration':10.,'eventType':'sz_foc'}])],ignore_index=True)
        y,_,_ = self.labels([100],events)
        self.assertEqual(y[0],-1)

    def test_missing_annotations_fail(self):
        with self.assertRaises(ValueError):
            read_events(None,600)

    def test_fractional_grid_rejected(self):
        with self.assertRaises(ValueError):
            integer_samples(.01,25)


class RealPilotTests(unittest.TestCase):
    """Integration checks against the downloaded pilot; never change raw files."""
    @classmethod
    def setUpClass(cls):
        cls.base = load_config()
        runs = BIDSLoader(cls.base['dataset']['raw_dir']).discover_runs()
        cls.recording = next((r for r in runs if r['subject']=='sub-125' and r['run']=='run-01'),None)
        if cls.recording is None:
            raise unittest.SkipTest('Local patient 125 pilot is not downloaded')

    def test_sensor_combinations_native_rates_offsets_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.base)
            c['dataset']['interim_dir'] = tmp
            c['preprocessing']['output_sampling_rate'] = None
            c['preprocessing']['window_sec'] = 10
            c['preprocessing']['stride_sec'] = 60
            c['preprocessing']['context_sec'] = 5
            for mods in (['ecg'],['ecg','eeg'],['ecg','eeg','mov']):
                c['sensors']['enabled'] = mods
                c['alignment']['offsets_sec']['ecg'] = 2
                report = StreamProcessor(c).process_run(self.recording)
                self.assertEqual(set(report['inventory']),set(mods))
                for mod in mods:
                    array = np.load(Path(report['signal_directory'])/f'{mod}.npy',mmap_mode='r')
                    self.assertEqual(array.shape[2],15*(25 if mod=='mov' else 256))
                    self.assertTrue(np.isfinite(array).all())
                    del array
                table = pd.read_csv(Path(report['result_directory'])/'windows.csv')
                self.assertEqual(table.prediction_time_sec.iloc[0],27)
                self.assertTrue((table.processing_support_start_sec>=2).all())
            c['labels']['minimum_warning_min'] = 20
            again = StreamProcessor(c).process_run(self.recording)
            self.assertTrue(again['signal_cache_reused'])
            self.assertEqual(report['signal_directory'],again['signal_directory'])
            self.assertNotEqual(report['result_directory'],again['result_directory'])

    def test_missing_channel_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.base)
            c['dataset']['interim_dir'] = tmp
            c['sensors']['channels']['eeg'] = ['NOT A CHANNEL']
            with self.assertRaisesRegex(ValueError,'available'):
                StreamProcessor(c).process_run(self.recording)

    def test_inconsistent_header_dates_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.base)
            c['dataset']['interim_dir'] = tmp
            c['alignment']['mode'] = 'header'
            with self.assertRaisesRegex(ValueError,'common time interval'):
                StreamProcessor(c).process_run(self.recording)


if __name__ == '__main__':
    unittest.main()

