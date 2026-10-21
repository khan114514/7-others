"""Reject unsupported solver budgets during validation before teacher/training."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'workspace/harbor_task/environment/engine'))
from method_contract import checked_spec


class SolverDomainTests(unittest.TestCase):
    def test_effective_solver_domain_matches_public_rule(self):
        for value in (1, 2, 129, True, 3.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checked_spec({'teacher': {'ode_nfe': value}})
        for value in (3, 10, 32, 128):
            self.assertEqual(checked_spec({'teacher': {'ode_nfe': value}}).teacher.ode_nfe, value)

    def test_all_archived_valid_calibration_configurations_unchanged(self):
        count = 0
        for file in (PKG / 'optimization_evidence/terminal_records').glob('run-*/method_config.json'):
            original = json.loads(file.read_text())['configuration']
            accepted = checked_spec({key: original[key] for key in
                                    ('name', 'teacher', 'time_sampler_note', 'train_notes')})
            self.assertEqual(asdict(accepted), original)
            count += 1
        self.assertGreaterEqual(count, 8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
