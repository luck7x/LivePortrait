"""Opt-in Linux tensor tests: skip BEFORE importing Torch/project tensor modules."""
import os
import sys
import unittest

if not sys.platform.startswith('linux') or os.environ.get('UPPER_ADAPTER_TENSOR_TESTS') != '1':
    raise unittest.SkipTest('Linux plus UPPER_ADAPTER_TENSOR_TESTS=1 required')

import copy
import io
import numpy as np
import torch
from src.modules.spade_generator import SPADEDecoder
from src.modules.upper_teeth_adapter import UpperTeethAdapterDecoder, safe_feature_mask
from src.utils.upper_teeth_support import safe_feature_mask as numpy_mask


class TensorContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = SPADEDecoder(upscale=2, max_features=16, block_expansion=4).eval()
        self.model = UpperTeethAdapterDecoder(self.base)
        self.feature = torch.rand(1, 16, 4, 4)
        self.s = torch.rand(1, 3, 16, 16)
        self.a = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
        self.a[..., 4:28, 4:28] = True
        self.v = self.a.float()

    def test_default_zero_gradient_mode_and_reload(self):
        before = {k: v.clone() for k, v in self.base.state_dict().items()}
        clone = copy.deepcopy(self.base)
        reference = clone(self.feature)
        self.model.train()
        self.assertFalse(any(m.training for m in self.base.modules()))
        self.assertTrue(torch.equal(reference, self.model(self.feature)))
        result = self.model(self.feature, self.s, self.v, self.a)
        self.assertTrue(torch.equal(reference, result))
        result.sum().backward()
        self.assertGreater(self.model.adapter[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.base.parameters()))
        with torch.no_grad():
            self.model.adapter[-1].weight.normal_(0, .03)
        changed = self.model(self.feature, self.s, self.v, self.a)
        self.assertFalse(torch.equal(changed, reference))
        self.assertTrue(torch.equal(changed[~self.a.expand_as(changed)], reference[~self.a.expand_as(reference)]))
        self.assertFalse(torch.equal(changed, self.model(self.feature, self.s*0, self.v, self.a)))
        for vis, mask in [(self.v*0, self.a), (self.v, self.a & False)]:
            self.assertTrue(torch.equal(reference, self.model(self.feature, self.s, vis, mask)))
        for k, v in self.base.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        stream = io.BytesIO()
        torch.save(self.model.adapter.state_dict(), stream)
        stream.seek(0)
        restored = UpperTeethAdapterDecoder(copy.deepcopy(self.base))
        restored.adapter.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(torch.equal(changed, restored(self.feature, self.s, self.v, self.a)))

    def test_support_oracle(self):
        for i in range(5):
            a = torch.rand(2, 1, 24, 32) > (.05*i)
            v = torch.rand(a.shape)
            v[..., 5:8, 9:11] = 0
            self.assertTrue(np.array_equal(safe_feature_mask(a, v).numpy(), numpy_mask(a.numpy(), v.numpy())))

    def test_validation_and_tail_rejection(self):
        with self.assertRaises(ValueError):
            UpperTeethAdapterDecoder(SPADEDecoder(upscale=1))
        for kw in ({'max_delta': .51}, {'max_delta': float('nan')}, {'width': 0}):
            with self.assertRaises(ValueError):
                UpperTeethAdapterDecoder(self.base, **kw)
        for s, v, a in [(self.s, None, self.a), (self.s.double(), self.v, self.a),
                        (self.s, self.v, self.a.float()), (self.s*float('nan'), self.v, self.a),
                        (self.s, self.v*2, self.a), (self.s[:, :2], self.v, self.a)]:
            with self.assertRaises(ValueError):
                self.model(self.feature, s, v, a)


if __name__ == '__main__':
    unittest.main()
