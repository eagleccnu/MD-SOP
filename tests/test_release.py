"""CPU-only tests of packaging/inference boundaries; no optimization or labels."""
from __future__ import annotations
import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from predict_features import checked_features,concatenate_features,load_model,predict_array,EXPERIMENTS
torch.set_num_threads(2)

class TestPortableRelease(unittest.TestCase):
    def test_dimensions_and_alignment(self):
        self.assertEqual(concatenate_features(np.zeros((4,1152),dtype=np.float32),np.zeros((4,30),dtype=np.float32)).shape,(4,1182))
        with self.assertRaises(ValueError):concatenate_features(np.zeros((4,1152)),np.zeros((3,30)))
        with self.assertRaises(ValueError):checked_features(np.zeros((4,1310)),1182)
        with self.assertRaises(ValueError):checked_features(np.zeros((0,1182)),1182)

    def test_nonfinite_and_integer_rejected(self):
        with self.assertRaises(TypeError):checked_features(np.zeros((4,1182),dtype=np.int64),1182)
        x=np.zeros((4,1182),dtype=np.float32);x[0,0]=np.nan
        with self.assertRaises(ValueError):checked_features(x,1182)

    def test_all_six_tensor_hashes_and_finite_predictions(self):
        array=np.random.default_rng(20261108).normal(size=(19,1182)).astype(np.float32)
        for exp in EXPERIMENTS:
            with self.subTest(exp=exp):
                model,metadata=load_model(exp);pred=predict_array(model,array)
                self.assertEqual(pred.shape,(19,));self.assertTrue(np.isfinite(pred).all())
                self.assertEqual(model.window_size,15 if exp.endswith('w15') else 1)
                expected=3504573 if exp.endswith('w15') else 2587069
                self.assertEqual(model.parameter_count(),expected)

    def test_split_and_concatenated_feature_predictions_match(self):
        array=np.random.default_rng(71).normal(size=(17,1182)).astype(np.float32)
        model,_=load_model('A_MD_w1');a=predict_array(model,array);b=predict_array(model,concatenate_features(array[:,:1152],array[:,1152:]))
        np.testing.assert_array_equal(a,b)

    def test_synthetic_example_is_explicitly_not_biological(self):
        root=Path(__file__).resolve().parents[1]
        array=np.load(root/'examples/synthetic_features.npy',allow_pickle=False)
        self.assertEqual(array.shape,(19,1182));self.assertTrue((root/'examples/README.md').is_file())
if __name__=='__main__':unittest.main()
