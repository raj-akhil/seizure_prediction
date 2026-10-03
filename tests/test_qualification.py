import json
from pathlib import Path
import tempfile
import unittest

from src.training.data import qualified_entries


class QualificationTests(unittest.TestCase):
    def test_unassigned_selected_subject_is_logged_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={'dataset': {'interim_dir': tmp},
                    'training': {'splits': {'train': ['sub-111'], 'validation': ['sub-112'], 'test': ['sub-113']}},
                    'qualification': {'enabled': True, 'exclude_unassigned_subjects': True,
                                      'continue_on_recording_error': True}}
            entries=[{'run': 'run-111', 'subject': 'sub-111', 'eligible': 12},
                     {'run': 'run-114', 'subject': 'sub-114', 'eligible': 7}]
            qualified=qualified_entries(config, entries)
            self.assertEqual([entry['run'] for entry in qualified], ['run-111'])
            report=json.loads((Path(tmp)/'qualification.json').read_text())
            self.assertEqual(report['excluded'][0]['id'], 'sub-114')
            self.assertEqual(report['excluded'][0]['reason'], 'not_assigned_to_training_partition')


if __name__ == '__main__':
    unittest.main()
