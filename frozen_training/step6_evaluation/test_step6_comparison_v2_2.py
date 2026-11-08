import csv
import gzip
from pathlib import Path
import tempfile
import unittest
import numpy as np
from step6_comparison_v2_2 import (parse_predictions,prediction_digest,score_predictions,write_scored_endpoint,
    grouped_pair_statistics,sign_flip_p,holm_adjust,paired_records)


class TestStep6Comparison(unittest.TestCase):
    def record(self):
        return {'record_id':'TOY','protein_id':'toy','sequence':'AAA','sequence_sha256':'0'*64,
            'joint_homology_group':'G','target':np.asarray([.1,0.,.9],dtype=np.float32),'qc_mask':np.asarray([True,False,True])}

    def test_raw_predictions_and_missing_labels(self):
        r=self.record();p={'TOY':np.asarray([-.2,3.,1.2])}
        per,summary=score_predictions(p,[r],{})
        self.assertAlmostEqual(per[0]['rmse'],.3,places=7)
        self.assertEqual(per[0]['out_of_range_fraction'],1.)
        r['target'][1]=np.nan
        again,_=score_predictions(p,[r],{});self.assertEqual(per[0]['rmse'],again[0]['rmse'])
        self.assertEqual(summary['eligible_residue_count'],2)

    def test_writer_roundtrip_and_role_identity(self):
        r=self.record();p={'TOY':np.asarray([-.2,.4,1.2])}
        identity={'model_version':'v2.2','run_tag':'v2_2_20261107','experiment_id':'TOY_MODEL',
            'endpoint':'md_test','checkpoint_file_sha256':'c'*64}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'out';write_scored_endpoint(root,p,[r],identity)
            back=parse_predictions(root/'per_residue_predictions.tsv.gz',[r],'TOY_MODEL','md_test','v2.2','c'*64)
            self.assertEqual(prediction_digest(p),prediction_digest(back))
            with self.assertRaises(AssertionError):parse_predictions(root/'per_residue_predictions.tsv.gz',[r],'WRONG','md_test')
            with self.assertRaises(AssertionError):write_scored_endpoint(root,p,[r],identity)

    def test_prediction_cannot_change_when_target_changes(self):
        r=self.record();p={'TOY':np.asarray([.2,.4,.8])};digest=prediction_digest(p)
        _,a=score_predictions(p,[r],{});r['target']=np.asarray([.7,0.,.6],dtype=np.float32)
        _,b=score_predictions(p,[r],{})
        self.assertEqual(digest,prediction_digest(p));self.assertNotEqual(a['protein_record_macro']['rmse'],b['protein_record_macro']['rmse'])

    def test_duplicate_positions_are_rejected(self):
        r=self.record();p={'TOY':np.asarray([.2,.4,.8])};identity={'experiment_id':'X','endpoint':'md_test'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'out';write_scored_endpoint(root,p,[r],identity)
            path=root/'per_residue_predictions.tsv.gz'
            with gzip.open(path,'rt') as f:text=f.read()
            lines=text.splitlines();lines[-1]=lines[-2]
            with gzip.open(path,'wt') as f:f.write('\n'.join(lines)+'\n')
            with self.assertRaises(AssertionError):parse_predictions(path,[r],'X','md_test')

    def test_group_bootstrap_distinguishes_record_and_group_weights(self):
        rows=[{'old_rmse':0.,'new_rmse':1.,'joint_homology_group':'a'},
              {'old_rmse':0.,'new_rmse':1.,'joint_homology_group':'a'},
              {'old_rmse':0.,'new_rmse':-1.,'joint_homology_group':'b'}]
        result=grouped_pair_statistics(rows,'rmse',20000,20261008)
        self.assertAlmostEqual(result['record_macro_delta'],1/3)
        self.assertEqual(result['group_balanced_delta'],0.)
        self.assertEqual(result['valid_homology_groups'],2)
        self.assertEqual(result['record_macro_group_bootstrap95_low'],-1.)
        self.assertEqual(result['record_macro_group_bootstrap95_high'],1.)

    def test_missing_correlation_is_not_zero(self):
        rows=[{'old_pcc':np.nan,'new_pcc':.2,'joint_homology_group':'a'},
              {'old_pcc':.3,'new_pcc':.5,'joint_homology_group':'b'}]
        result=grouped_pair_statistics(rows,'pcc',200,1)
        self.assertEqual(result['valid_record_pairs'],1);self.assertAlmostEqual(result['record_macro_delta'],.2)

    def test_sign_flip_exact_and_monte_carlo(self):
        p,method,n=sign_flip_p([1.,1.],20000,1);self.assertEqual(p,.5);self.assertEqual(method,'exact');self.assertEqual(n,4)
        p,method,n=sign_flip_p(np.zeros(20),20000,1);self.assertEqual(p,1.);self.assertEqual(method,'monte_carlo_plus_one')

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])

    def test_paired_direction_and_absolute_bias(self):
        r=self.record();a,_=score_predictions({'TOY':np.asarray([.2,0.,1.])},[r],{})
        b,_=score_predictions({'TOY':np.asarray([.1,0.,.9])},[r],{})
        pairs=paired_records(a,b,'X','md_test')
        self.assertLess(pairs[0]['new_minus_old_rmse'],0)
        self.assertLess(pairs[0]['new_minus_old_abs_bias'],0)


if __name__=='__main__':unittest.main()
