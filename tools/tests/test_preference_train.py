import unittest
from tools.tests.dependencies import require
require('torch','regex','numpy')
import torch
import math
from tools.correction.preference_data import edit_weights
from tools.correction.preference_train import preference_loss
from tools.correction.byte_data import encode

class PreferenceTests(unittest.TestCase):
    def test_grapheme_weights_align_with_all_utf8_bytes_and_eos(self):
        for left,right in [('আমি যাই।','আমি যায়।'),('ক',''),('','ক'),('আমি','আমি'),('ক্ষ','ক')]:
            w=edit_weights(left,right);self.assertEqual(len(w),len(encode(left)))
            self.assertTrue(all(v in (1.,2.,4.) for v in w))
            if left!=right:self.assertIn(4.,w)
        self.assertEqual(edit_weights('আমি','আমি'),[1.]*len(encode('আমি')))
        self.assertEqual(edit_weights('ক্ষ','ক')[:-1],[4.]*len('ক্ষ'.encode()))
    def test_preference_gradient_rewards_chosen_and_penalizes_rejected(self):
        c=torch.tensor([-2.],requires_grad=True);r=torch.tensor([-3.],requires_grad=True)
        loss,advantage=preference_loss(c,r,c.detach(),r.detach())
        self.assertAlmostEqual(float(loss),math.log(2),places=6);self.assertEqual(float(advantage),0.)
        loss.backward();self.assertLess(float(c.grad),0);self.assertGreater(float(r.grad),0)
        better,_=preference_loss(c.detach()+1,r.detach()-1,c.detach(),r.detach())
        self.assertLess(float(better),float(loss))
