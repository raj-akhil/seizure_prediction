"""Small estimator tests; separate from scientific performance evaluation."""
import unittest
import numpy as np
import pandas as pd
from src.models.selection import FeatureSelection


class FittedSelectionTests(unittest.TestCase):
    def test_fit_statistics_do_not_change_on_unseen_data(self):
        x=pd.DataFrame({'eeg::a::rms':[1.,2.,3.,4.,5.,6.],
                        'ecg::b::mean':[2.,np.nan,4.,8.,3.,9.]})
        selector=FeatureSelection(k='all').fit(x,[0,0,0,1,1,1])
        before=selector.medians_.copy()
        result=selector.transform(pd.DataFrame({'eeg::a::rms':[1e9],'mov::unknown::rms':[7.]}))
        pd.testing.assert_series_equal(before,selector.medians_)
        self.assertTrue(np.isfinite(result).all())
        self.assertFalse(any('unknown' in name for name in selector.get_feature_names_out()))

    def test_missing_constant_and_duplicate_features_pruned(self):
        x=pd.DataFrame({'eeg::a::rms':np.arange(10,dtype=float),
                        'eeg::a::duplicate':np.arange(10,dtype=float)*2,
                        'ecg::b::constant':np.ones(10),
                        'mov::c::absent':np.full(10,np.nan)})
        selector=FeatureSelection(k='all').fit(x,[0]*5+[1]*5)
        self.assertEqual(selector.get_feature_names_out().tolist(),['eeg::a::rms'])

    def test_all_missing_training_fold_refused(self):
        with self.assertRaisesRegex(ValueError,'No usable features'):
            FeatureSelection().fit(pd.DataFrame({'eeg::a::rms':[np.nan]*4}),[0,0,1,1])


if __name__=='__main__':
    unittest.main()
