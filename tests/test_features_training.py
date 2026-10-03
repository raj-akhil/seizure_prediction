import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.features.signal_features import channel_features, cardiac_features
from src.models.training import preflight, candidates
from src.models.selection import FeatureSelection
from src.models.evaluation import alarm_metrics, choose_threshold


class FeatureTests(unittest.TestCase):
    def test_known_sine_spectrum_and_rms(self):
        fs=256;t=np.arange(1280)/fs
        x=np.sin(2*np.pi*10*t)[None,None,:]
        f=channel_features(x,fs,fs,['time','spectral','hjorth','entropy'],{'alpha':[8,13],'delta':[.5,4]},2)
        self.assertAlmostEqual(f['rms'][0,0],np.sqrt(.5),places=6)
        self.assertEqual(f['dominant_frequency_hz'][0,0],10)
        self.assertGreater(f['alpha_relative_power'][0,0],.99)
        self.assertLess(f['delta_relative_power'][0,0],.001)

    def test_upsampling_does_not_create_supported_frequency_band(self):
        x=np.sin(2*np.pi*2*np.arange(1280)/256)[None,None,:]
        f=channel_features(x,256,25,['spectral'],{'impossible':[15,30]},2)
        self.assertTrue(np.isnan(f['impossible_power']).all())

    def test_known_heart_rate_and_polarity(self):
        cfg=load_config()['features']['ecg'];fs=256
        t=np.arange(60*fs)/fs
        x=np.zeros_like(t)
        for beat in range(1,60):
            x+=np.exp(-((t-beat)/.015)**2)
        for sign in (1,-1):
            f=cardiac_features(sign*x,fs,cfg)
            self.assertAlmostEqual(f['heart_rate_bpm'],60,delta=1)
            self.assertEqual(f['hrv_available'],1)
            self.assertLess(f['rr_rmssd_ms'],5)

    def test_short_ecg_does_not_claim_hrv(self):
        cfg=load_config()['features']['ecg']
        f=cardiac_features(np.sin(np.arange(1280)),256,cfg)
        self.assertEqual(f['hrv_available'],0)
        self.assertTrue(np.isnan(f['rr_sd_ms']))


class TrainingNoFitTests(unittest.TestCase):
    def fixture(self):
        cfg=load_config()
        cfg['training']['splits']={'train':['a','b','c'],'validation':['d'],'test':['e']}
        cfg['training']['min_positive_events_per_partition']=1
        rows=[]
        for subject in 'abcde':
            for y in (0,1,0,1):
                rows.append({'subject':subject,'label':y,'recording_id':subject,'event_row':0})
        return cfg,pd.DataFrame({'eeg::channel::rms':np.arange(20)}),pd.DataFrame(rows)

    def test_preflight_keeps_subjects_separate_without_fit(self):
        cfg,x,meta=self.fixture()
        with patch.object(FeatureSelection,'fit',side_effect=AssertionError('Fitting forbidden')):
            indices,folds=preflight(cfg,x,meta)
            self.assertEqual(len(folds),3)
            self.assertFalse(set(indices['train'])&set(indices['test']))
            self.assertEqual(set(candidates(cfg)),{'logistic','hist_gradient_boosting'})
            self.assertFalse(candidates(cfg)['hist_gradient_boosting'][0]['model'].early_stopping)

    def test_overlap_and_single_subject_refused(self):
        cfg,x,meta=self.fixture()
        cfg['training']['splits']['test']=['a']
        with self.assertRaisesRegex(ValueError,'overlap'):
            preflight(cfg,x,meta)
        cfg['training']['splits']={'train':[],'validation':[],'test':[]}
        with self.assertRaisesRegex(ValueError,'explicit'):
            preflight(cfg,x,meta)

    def test_missing_channels_align_by_name_using_fixed_statistics(self):
        # Hand-set statistics test transformation only; no fit/selection is run.
        s=FeatureSelection(add_missing_indicators=True)
        s.columns_=['eeg::a::rms','ecg::b::rms']
        s.medians_=pd.Series([2.,4.],index=s.columns_)
        s.means_=s.medians_.copy();s.scales_=pd.Series([1.,2.],index=s.columns_)
        s.indices_=np.arange(4)
        values=s.transform(pd.DataFrame({'ecg::b::rms':[8.],'mov::unused::rms':[100.]}))
        np.testing.assert_array_equal(values,[[0.,2.,1.,0.]])

    def test_alarm_episodes_and_refractory(self):
        cfg=load_config()
        cfg['labels'].update(minimum_warning_min=1,occurrence_period_min=1)
        cfg['training']['alarms'].update(consecutive_windows=2,refractory_sec=120)
        frame=pd.DataFrame({'recording_id':['r']*4,'prediction_time_sec':[65,70,75,80]})
        events={'r':pd.DataFrame({'onset':[180.],'duration':[5.],'eventType':['sz_foc']})}
        r=alarm_metrics(frame,[.9]*4,.5,cfg,events)
        self.assertEqual(r['alarms'],1)
        self.assertEqual(r['eligible_events'],1)
        self.assertEqual(r['event_sensitivity'],1)
        self.assertEqual(r['false_alarms'],0)

    def test_alarm_streak_resets_across_gap(self):
        cfg=load_config()
        cfg['training']['alarms']['consecutive_windows']=2
        frame=pd.DataFrame({'recording_id':['r','r'],'prediction_time_sec':[5,20]})
        events={'r':pd.DataFrame({'onset':[9999.],'duration':[5.],'eventType':['sz_foc']})}
        self.assertEqual(alarm_metrics(frame,[1.,1.],.5,cfg,events)['alarms'],0)


if __name__=='__main__':
    unittest.main()
