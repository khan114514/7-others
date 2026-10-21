"""Regression tests for scoring and the typed candidate boundary."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'environment/engine'))
from method_contract import checked_spec, checked_times, checked_indices, checked_weights
from scoring import normalize


class BoundaryTests(unittest.TestCase):
    def test_unclipped_and_finite_reward(self):
        anchors = {'calibration_status':'FROZEN', 'baseline':10., 'upper_bound':5., 'direction':'minimize'}
        self.assertEqual(normalize(0., anchors), 2.)
        self.assertEqual(normalize(15., anchors), -1.)
        for value in (float('nan'), float('inf'), True, -1):
            with self.assertRaises(ValueError): normalize(value, anchors)
        with self.assertRaises(ValueError): normalize(9., dict(anchors, calibration_status='PENDING'))

    def test_reject_sampler_shape_range_and_nonfinite(self):
        for values in ([.1], [.1, float('nan')], [.1, 1.1], [.1, True]):
            with self.assertRaises(ValueError): checked_times(values, 2)
        self.assertEqual(checked_times([0., 1.], 2), [0., 1.])

    def test_filter_cannot_cross_labels(self):
        self.assertEqual(checked_indices([1, 2], 2, 2), [1, 2])
        for values in ([2, 3], [0, 0], [0.0, 2], [0]):
            with self.assertRaises(ValueError): checked_indices(values, 2, 2)

    def test_config_cannot_change_initialization_or_import_assets(self):
        self.assertEqual(checked_spec({'teacher':{}}).initialization, 'warmstart_official_dd_ema')
        for value in ({'teacher':{}, 'initialization':'weakened'},
                      {'teacher':{'source_manifest':'other-track'}},
                      {'teacher':{'teacher_steps':1}}, {'teacher':{'num_samples':1}},
                      {'teacher':{'guidance':float('nan')}}, {'teacher':{'reuse':True}}):
            with self.assertRaises(ValueError): checked_spec(value)

    def test_weights_preserve_scale_and_reject_invalid_mass(self):
        self.assertEqual(checked_weights([1.,3.],2), [.5,1.5])
        for values in ([0.,0.], [-1.,2.], [float('nan'),1.], [True,1.], [17.,1.]):
            with self.assertRaises(ValueError): checked_weights(values,2)


if __name__ == '__main__':
    unittest.main()
