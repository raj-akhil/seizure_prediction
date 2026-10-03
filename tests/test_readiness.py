import copy
import unittest

import pandas as pd

from src.utils.config import load_config
from src.training.readiness import report, repair_splits


class ReadinessRepairTests(unittest.TestCase):
    def fixture(self):
        config=load_config()
        config['training']['splits']={'train':['b','c','d','e'],'validation':['a'],'test':['f']}
        rows=[]
        # 'a' has only positive windows; 'e' has only background windows.
        for subject,negative,events in [('a',0,2),('b',10,2),('c',10,2),('d',10,1),('e',10,0),('f',10,1)]:
            for i in range(negative+events):
                rows.append({'subject':subject,'recording_id':subject,'prediction_time_sec':i*5,
                             'label':int(i>=negative),'event_row':i-negative if i>=negative else None})
        return config,pd.DataFrame(rows)

    def test_reports_all_partition_failures_and_exact_counts(self):
        config,meta=self.fixture()
        result=report(config,meta)
        self.assertFalse(result['ready'])
        self.assertIn('validation needs both classes: 0 background',result['issues'][0])
        self.assertIn('test has only 1 positive seizure episodes; requires 2',result['issues'][0])
        self.assertEqual(result['partitions']['validation']['positive_windows'],2)

    def test_repairs_groups_without_relaxing_event_or_fold_requirements(self):
        config,meta=self.fixture();original=copy.deepcopy(config)
        result=repair_splits(config,meta)
        self.assertTrue(result['changed'])
        self.assertTrue(result['after']['ready'])
        self.assertEqual(len(result['after']['inner_folds']),3)
        for partition in result['after']['partitions'].values():
            self.assertGreater(partition['background_windows'],0)
            self.assertGreaterEqual(partition['positive_events'],2)
        changed=copy.deepcopy(result['config'])
        changed['training']['splits']=original['training']['splits']
        self.assertEqual(changed,original)
        self.assertEqual(config,original)
        self.assertEqual(result['config']['training']['splits'],repair_splits(config,meta)['config']['training']['splits'])

    def test_insufficient_data_fails_without_silent_parameter_changes(self):
        config,meta=self.fixture()
        config['training']['min_positive_events_per_partition']=10
        with self.assertRaisesRegex(ValueError,'insufficient eligible seizure episodes'):
            repair_splits(config,meta)
        self.assertEqual(config['training']['min_positive_events_per_partition'],10)

    def test_valid_split_is_left_unchanged(self):
        config,meta=self.fixture()
        fixed=repair_splits(config,meta)['config']
        result=repair_splits(fixed,meta)
        self.assertFalse(result['changed'])
        self.assertEqual(result['config'],fixed)
