import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from step7_experimental_comparison_v2_2 import (load_legacy_predictions,paired_comparisons,score_predictions,
    atomic_write_gzip_tsv,sign_flip_p,holm_adjust)


class TestExperimentalComparison(unittest.TestCase):
    def record(self):
        return {'record_id':'SYNTHETIC','protein_id':'toy','chain_id':'A','sequence':'AAA','sequence_sha256':'a'*64,
            'joint_homology_group':'G0','target':np.asarray([.2,0.,.8],dtype=np.float32),'qc_mask':np.asarray([True,False,True])}

    def rows(self,r):
        return [{'model_id':'legacy','record_id':r['record_id'],'protein_id':r['protein_id'],'chain_id':r['chain_id'],
            'joint_homology_group':r['joint_homology_group'],'sequence_sha256':r['sequence_sha256'],
            'residue_index_1based':j+1,'amino_acid':'A','qc_eligible':bool(r['qc_mask'][j]),
            'target':float(r['target'][j]) if r['qc_mask'][j] else None,'prediction':[-.2,2.,1.3][j]} for j in range(3)]

    def load(self,r,rows,models=('legacy',)):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'toy.tsv.gz';atomic_write_gzip_tsv(path,rows,list(rows[0]))
            return load_legacy_predictions(path,[r],models)

    def test_legacy_raw_values_and_order(self):
        r=self.record();rows=list(reversed(self.rows(r)));out=self.load(r,rows)
        np.testing.assert_array_equal(out['legacy']['SYNTHETIC'],[-.2,2.,1.3])

    def test_changed_mask_rejected(self):
        r=self.record();rows=self.rows(r);rows[1]['qc_eligible']=True
        with self.assertRaises(AssertionError):self.load(r,rows)

    def test_wrong_chain_and_sequence_rejected(self):
        r=self.record();rows=self.rows(r);rows[0]['chain_id']='B'
        with self.assertRaises(AssertionError):self.load(r,rows)
        rows=self.rows(r);rows[0]['amino_acid']='C'
        with self.assertRaises(AssertionError):self.load(r,rows)

    def test_wrong_group_and_target_rejected(self):
        r=self.record();rows=self.rows(r);rows[0]['joint_homology_group']='OTHER'
        with self.assertRaises(AssertionError):self.load(r,rows)
        rows=self.rows(r);rows[0]['target']=.4
        with self.assertRaises(AssertionError):self.load(r,rows)

    def test_missing_and_duplicate_positions_rejected(self):
        r=self.record();rows=self.rows(r)
        with self.assertRaises(AssertionError):self.load(r,rows[:-1])
        rows[2]=dict(rows[1])
        with self.assertRaises(AssertionError):self.load(r,rows)

    def test_unregistered_model_rejected(self):
        r=self.record()
        with self.assertRaises(AssertionError):self.load(r,self.rows(r),('different',))

    def test_eight_group_exact_p_and_conservative_family(self):
        p,method,n=sign_flip_p(np.ones(8),20000,20261107)
        self.assertEqual(method,'exact');self.assertEqual(n,256);self.assertEqual(p,2/256)
        self.assertEqual(min(holm_adjust([p]+[1.]*23)),.1875)

    def test_prespecified_24_comparisons_and_168_statistics(self):
        experiments=['A_MD_w1','A_MD_w15','B_SOPPE_w1','B_SOPPE_w15','C_MIX_w1','C_MIX_w15']
        historical=['historical_original_soppcl','historical_mu64_bilstm','historical_raw1310_bilstm']
        records=[]
        for i in range(10):
            r=self.record();r.update({'record_id':'TOY'+str(i),'protein_id':'p'+str(i),'joint_homology_group':'G'+str(max(0,i-2))});records.append(r)
        scores={}
        models=['v2_2__'+e for e in experiments]+['v2_1__'+e for e in experiments]+historical
        for i,model in enumerate(models):
            pred={r['record_id']:r['target'].astype(np.float64)+.01*(i+1) for r in records}
            scores[model]=score_predictions(pred,records,{'model_id':model})
        policy={'bootstrap_replicates':20000,'bootstrap_seed':20261008,'sign_flip_replicates':20000,
            'sign_flip_seed':20261107,'exact_sign_flip_max_groups':16}
        pairs,stats=paired_comparisons(scores,experiments,historical,policy)
        self.assertEqual(len(pairs),240);self.assertEqual(len(stats),168)
        self.assertEqual(len([r for r in stats if r['metric']=='rmse']),24)
        self.assertTrue(all(r['valid_homology_groups']==8 for r in stats))


if __name__=='__main__':unittest.main()
