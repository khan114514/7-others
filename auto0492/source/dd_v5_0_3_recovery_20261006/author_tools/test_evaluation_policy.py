"""Verify that formal scoring cannot silently reuse public seeds or anchors."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ENGINE=Path(__file__).resolve().parent.parent/'workspace/harbor_task/environment/engine'
sys.path.insert(0,str(ENGINE))
from evaluation_policy import load_formal_policy


class HeldOutPolicy(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.task=self.root/'task';self.task.mkdir()
        self.policy={'protocol':'dd-selfproto-v1.3','sample_count':50000,'evaluation_seed':134217729,
                     'teacher_seed':1001,'training_seed':2001,
                     'anchors':{'calibration_status':'FROZEN','direction':'minimize',
                                'baseline':10.0,'upper_bound':9.0,'evaluation_seed':134217729},'evidence':{}}
        for name,metric in [('baseline',10.0),('upper_bound',9.0)]:
            p=self.root/(name+'.json');p.write_text(json.dumps({'status':'SUCCESS','sample_count':50000,
                  'evaluation_seed':134217729,'metric':metric}))
            self.policy['evidence'][name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        self.path=self.root/'private.json';self.write()
        self.config={'formal_evaluation_policy':str(self.path)}

    def tearDown(self):self.directory.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.policy));self.path.chmod(0o600)

    def load(self):return load_formal_policy(self.config,self.task,'dd-selfproto-v1.3')

    def test_valid_empirical_held_out_policy_and_hash_binding(self):
        p,binding=self.load();self.assertEqual(p['anchors']['baseline'],10)
        self.assertEqual(binding,hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.policy['evidence']['baseline']['sha256']='0'*64;self.write()
        with self.assertRaisesRegex(ValueError,'receipt changed'):self.load()

    def test_public_seed_and_unpaired_anchors_are_rejected(self):
        for seed in [0,True,-1,2**31]:
            self.policy['evaluation_seed']=seed;self.write()
            with self.subTest(seed=seed),self.assertRaisesRegex(ValueError,'held-out seed'):self.load()
        self.policy['evaluation_seed']=134217729
        self.policy['anchors']['evaluation_seed']=1;self.write()
        with self.assertRaisesRegex(ValueError,'held-out seed'):self.load()

    def test_receipt_content_must_be_complete_and_same_seed(self):
        receipt=self.root/'baseline.json'
        for alteration in [{'sample_count':64},{'status':'FAILED'},{'metric':10.1},{'evaluation_seed':0}]:
            contents={'status':'SUCCESS','sample_count':50000,'evaluation_seed':134217729,'metric':10.0,**alteration}
            receipt.write_text(json.dumps(contents))
            self.policy['evidence']['baseline']['sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest();self.write()
            with self.subTest(alteration=alteration),self.assertRaisesRegex(ValueError,'binding mismatch'):self.load()

    def test_private_file_cannot_be_in_task_or_world_readable(self):
        self.path.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'permissions'):self.load()
        self.path=self.task/'private.json';self.write();self.config['formal_evaluation_policy']=str(self.path)
        with self.assertRaisesRegex(ValueError,'outside the delivered task'):self.load()

    def test_missing_policy_and_changed_training_protocol_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'not configured'):load_formal_policy({},self.task,'dd-selfproto-v1.3')
        self.policy['training_seed']=2002;self.write()
        with self.assertRaisesRegex(ValueError,'published streams'):self.load()


if __name__=='__main__':unittest.main()
