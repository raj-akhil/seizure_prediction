import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.validation.metrics import window_metrics, calibration_table
from src.validation.report import write_report, bootstrap, settings
from src.models.evaluation import alarm_metrics


class ValidationTests(unittest.TestCase):
    def fixture(self):
        c=load_config()
        c['labels'].update(minimum_warning_min=1,occurrence_period_min=1)
        c['training']['alarms'].update(consecutive_windows=1,refractory_sec=120)
        c['validation'].update(plots=False,bootstrap_replicates=20)
        f=pd.DataFrame({'subject':['a']*4,'recording_id':['r']*4,
            'prediction_time_sec':[65.,70.,75.,80.],'label':[1]*4,'probability':[.9]*4})
        e={'r':pd.DataFrame({'onset':[180.,500.],'duration':[5.,5.],'eventType':['sz_foc']*2})}
        return c,f,e

    def test_known_confusion_and_skill(self):
        m=window_metrics([0,0,1,1],[.1,.6,.4,.9],.5,.5)
        self.assertEqual([m[k] for k in ['tn','fp','fn','tp']],[1]*4)
        self.assertEqual(m['f1'],.5)
        self.assertAlmostEqual(m['brier_score'],.185)
        self.assertAlmostEqual(m['brier_skill_score'],.26)

    def test_no_skill_baseline_and_single_class(self):
        m=window_metrics([0,0,0,1],[.25]*4,.5,.25)
        self.assertEqual(m['brier_skill_score'],0)
        self.assertEqual(m['average_precision'],.25)
        self.assertIsNone(m['precision'])
        m=window_metrics([0,0],[0,0],.5,0)
        self.assertIsNone(m['roc_auc'])
        self.assertIsNone(m['brier_skill_score'])

    def test_invalid_probabilities_and_empty_inputs(self):
        for y,p in [([],[]),([1],[np.nan]),([0],[1.2]),([-1],[.1]),([1,0],[.5])]:
            with self.assertRaises(ValueError):
                window_metrics(y,p)

    def test_calibration_endpoints_and_empty_bins(self):
        bins=calibration_table([0,1],[0.,1.],10)
        self.assertEqual(bins[0]['count'],1)
        self.assertEqual(bins[-1]['count'],1)
        self.assertIsNone(bins[2]['observed_fraction'])

    def test_event_deduplication_and_unobservable_event(self):
        c,f,e=self.fixture()
        a=alarm_metrics(f,f.probability,.5,c,e)
        self.assertEqual(a['warned_events'],1)
        self.assertEqual(a['unevaluable_events_in_evaluated_runs'],1)
        self.assertEqual(a['median_first_warning_lead_sec'],115)
        self.assertEqual(a['time_in_warning_fraction_eligible'],1)

    def test_half_open_target_interval(self):
        c,f,e=self.fixture()
        f=f.iloc[:1].copy();f['prediction_time_sec']=60
        self.assertEqual(alarm_metrics(f,[1.],.5,c,e)['warned_events'],0) # onset == upper boundary
        f['prediction_time_sec']=120
        self.assertEqual(alarm_metrics(f,[1.],.5,c,e)['warned_events'],1) # onset == lower boundary

    def test_overlap_exposure_and_partial_warning(self):
        c,f,e=self.fixture()
        c['preprocessing'].update(window_sec=10,stride_sec=5)
        c['labels'].update(minimum_warning_min=0,occurrence_period_min=7/60)
        a=alarm_metrics(f,[1,0,0,0],.5,c,e)
        self.assertAlmostEqual(a['eligible_monitoring_hours'],20/3600)
        self.assertAlmostEqual(a['time_in_warning_fraction_eligible'],7/20)
        self.assertEqual(a['false_alarms'],1)

    def test_gap_is_not_counted_as_monitoring(self):
        c,f,e=self.fixture()
        f=f.iloc[:2].copy();f['prediction_time_sec']=[65,1000]
        a=alarm_metrics(f,[1,0],.5,c,e)
        self.assertAlmostEqual(a['eligible_monitoring_hours'],10/3600)
        self.assertEqual(a['time_in_warning_fraction_eligible'],.5)

    def test_duplicate_decisions_and_invalid_probabilities_rejected(self):
        c,f,e=self.fixture()
        with self.assertRaises(ValueError):
            alarm_metrics(f,[np.nan]*4,.5,c,e)
        f.loc[1,'prediction_time_sec']=65
        with self.assertRaisesRegex(ValueError,'unique'):
            alarm_metrics(f,f.probability,.5,c,e)

    def test_report_and_insufficient_patient_ci(self):
        c,f,e=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'report'
            report=write_report(f,c,e,.5,.2,dest,'synthetic')
            self.assertTrue((dest/'metrics.csv').exists())
            self.assertIn('not_computed',report['uncertainty']['status'])
            f.loc[0,'label']=0
            with self.assertRaisesRegex(ValueError,'labels'):
                write_report(f,c,e,.5,.2,Path(tmp)/'bad','synthetic')

    def test_cluster_bootstrap_is_reproducible(self):
        c,f,e=self.fixture()
        frames=[];alarms={}
        for i in range(5):
            b=f.copy();b['subject']=str(i);b['label']=[0,0,1,1]
            b['probability']=[.1,.3,.7,.9];frames.append(b)
            alarms[str(i)]=alarm_metrics(f,f.probability,.5,c,e)
        combined=pd.concat(frames,ignore_index=True)
        a=bootstrap(combined,alarms,.5,.5,settings(c))
        self.assertEqual(a,bootstrap(combined,alarms,.5,.5,settings(c)))
        self.assertEqual(a['intervals']['average_precision']['lower'],1)
        self.assertEqual(a['intervals']['event_sensitivity']['valid_replicates'],20)


if __name__=='__main__':
    unittest.main()
