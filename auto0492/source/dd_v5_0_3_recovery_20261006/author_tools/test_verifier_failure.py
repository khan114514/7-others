"""Exercise the real shell entry with relocated test-only absolute paths."""
from pathlib import Path
import subprocess
import tempfile
import unittest

TASK = Path(__file__).resolve().parents[1] / 'workspace/harbor_task'


class VerifierFailureTests(unittest.TestCase):
    def execute(self, source):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        output = root / 'verifier'
        output.mkdir()
        (output / 'reward.txt').write_text('999\n')
        (output / 'result.json').write_text('{"stale":true}')
        (output / 'invocation.json').write_text('{"stale":true}')
        (output / 'failure.json').write_text('{"failure_category":"HARD_GATE","stale":true}')
        client = root / 'client.py'
        client.write_text(source)
        original = (TASK / 'tests/test.sh').read_text()
        relocated = original.replace('/logs/verifier', str(output)).replace(
            '/tests/runtime_client.py', str(client))
        entry = root / 'test.sh'
        entry.write_text(relocated)
        result = subprocess.run(['bash', str(entry)], capture_output=True, text=True)
        return result, output

    def test_infrastructure_failure_never_fabricates_baseline_zero(self):
        result, output = self.execute('raise SystemExit(2)\n')
        self.assertEqual(result.returncode, 2)
        self.assertFalse((output / 'reward.txt').exists())
        self.assertFalse((output / 'result.json').exists())
        self.assertFalse((output / 'failure.json').exists())

    def test_partial_outputs_removed_but_fresh_invocation_preserved(self):
        result, output = self.execute(
            "import argparse,json\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--out');args,_=p.parse_known_args()\n"
            "out=Path(args.out)\n"
            "(out/'invocation.json').write_text('{\"fresh\":true}')\n"
            "(out/'reward.txt').write_text('1.0\\n')\n"
            "(out/'result.json.stage').write_text('{}')\n"
            "raise SystemExit(3)\n")
        self.assertEqual(result.returncode, 3)
        self.assertFalse((output / 'reward.txt').exists())
        self.assertFalse((output / 'result.json.stage').exists())
        self.assertEqual((output / 'invocation.json').read_text(), '{"fresh":true}')

    def test_successful_baseline_zero_remains_a_valid_score(self):
        result, output = self.execute(
            "import argparse\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--out');args,_=p.parse_known_args()\n"
            "out=Path(args.out)\n(out/'reward.txt').write_text('0\\n')\n"
            "(out/'result.json').write_text('{\"status\":\"SUCCESS\",\"reward\":0}')\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual((output / 'reward.txt').read_text(), '0\n')
        self.assertTrue((output / 'result.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
