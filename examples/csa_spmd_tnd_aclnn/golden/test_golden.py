"""CPU small-shape self tests; these do NOT certify production CSA accuracy."""
import unittest
import torch
from golden import Config, compressor_step, csa_golden, hc_post, int8_mm, quantize, rms, rope


def fixture(tokens=8):
    torch.manual_seed(1024)
    cfg = Config(heads=2, index_heads=2, head_dim=8, index_dim=4, rope_dim=2,
                 groups=2, window=4, topk=3)
    def rand(*shape):
        return torch.randn(*shape) * 0.1
    w = {'hc_attn_fn': rand(24, 32), 'hc_attn_scale': torch.ones(3), 'hc_attn_base': rand(24),
         'attn_norm_w': torch.ones(8), 'wq_a': rand(4, 8), 'gamma_cq': torch.ones(4),
         'wq_b': torch.randint(-10, 11, (4, 16), dtype=torch.int8), 'wq_b_scale': torch.ones(16) * .02,
         'wkv': rand(8, 8), 'gamma_ckv': torch.ones(8),
         'idx_wq_b': torch.randint(-10, 11, (4, 8), dtype=torch.int8), 'idx_wq_b_scale': torch.ones(8) * .02,
         'weights_proj': rand(8, 2), 'attn_sink': rand(2), 'wo_a': rand(2, 8, 4),
         'wo_b': torch.randint(-10, 11, (8, 8), dtype=torch.int8), 'wo_b_scale': torch.ones(8) * .02,
         'hadamard_idx': torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]).float()}
    for prefix, dim in [('cmp', 8), ('inner', 4)]:
        w.update({prefix+'_wkv': rand(2*dim, 8), prefix+'_wgate': rand(2*dim, 8),
                  prefix+'_ape': rand(4, 2*dim), prefix+'_norm_w': torch.ones(dim)})
    r = {'history': 0, 'query_length': tokens, 'swa': torch.empty(0,8),
         'compressed': torch.empty(0,8), 'index_keys': torch.empty(0,4,dtype=torch.int8),
         'index_scales': torch.empty(0,1)}
    for name, dim in [('main_state',8), ('index_state',4)]:
        r[name] = {'positions': [], 'values': torch.empty(0,2*dim), 'logits': torch.empty(0,2*dim)}
    angles = torch.arange(32).float()[:,None].repeat(1,2) * .03
    return {'x': rand(tokens,4,8).bfloat16(), 'weights': w, 'requests': [r],
            'rope_cos': angles.cos(), 'rope_sin': angles.sin(), 'cfg': cfg}


class Tests(unittest.TestCase):
    def test_int8_mm_exact(self):
        torch.manual_seed(1)
        a = torch.randint(-128,128,(3,1025),dtype=torch.int8)
        b = torch.randint(-128,128,(1025,5),dtype=torch.int8)
        self.assertTrue(torch.equal(int8_mm(a,b).long(), a.long() @ b.long()))

    def test_zero_quant(self):
        q,s = quantize(torch.zeros(2,4))
        self.assertTrue(torch.isfinite(s).all())
        self.assertFalse(q.any())

    def test_compressor_overlap_formula(self):
        cfg = Config()
        values = torch.arange(8 * 4).reshape(8, 4).float()
        state = {'positions': list(range(7)), 'values': values[:7], 'logits': torch.zeros(7,4)}
        weight = values[-1:, :].T
        result, updated = compressor_step(torch.ones(1), 7, state, weight, torch.zeros(4,1),
                                          torch.zeros(4,4), torch.ones(2), torch.ones(2), torch.zeros(2), cfg)
        expected_pool = torch.cat((values[:4,:2],values[4:,2:]),0).mean(0)
        torch.testing.assert_close(result, rms(expected_pool).bfloat16())
        self.assertEqual(updated['positions'], list(range(1,8)))

    def test_global_quant_differs_from_group_quant(self):
        # Regression: native wo_b scale covers the flattened O_A row.
        x = torch.tensor([[[1., 2.], [64., 127.]]])
        _, group_scale = quantize(x)
        q, row_scale = quantize(x.flatten(1))
        self.assertEqual(row_scale.shape, (1,1))
        self.assertEqual(row_scale.item(), 1.)
        self.assertNotEqual(group_scale[0,0,0].item(), row_scale.item())
        self.assertTrue(torch.equal(q, torch.tensor([[1,2,64,127]],dtype=torch.int8)))

    def test_future_tokens_not_visible(self):
        f = fixture()
        baseline = csa_golden(**f)
        f['x'][5:] = (f['x'][5:].float() * 10).bfloat16()
        changed = csa_golden(**f)
        torch.testing.assert_close(baseline['output'][:5],changed['output'][:5],atol=0,rtol=0)

    def test_rope_inverse(self):
        x = torch.randn(2,8)
        c,s = torch.full((2,),.7).cos(), torch.full((2,),.7).sin()
        torch.testing.assert_close(rope(rope(x,c,s),c,s,True),x)

    def test_hc_orientation(self):
        residual = torch.arange(12).reshape(1,4,3).bfloat16()
        comb = torch.eye(4).roll(1,1)[None]
        out = hc_post(torch.zeros(1,3),residual,torch.zeros(1,4),comb)
        torch.testing.assert_close(out, residual.roll(1,1))

    def test_full_chain_and_no_mutation(self):
        f = fixture()
        x0 = f['x'].clone()
        a = csa_golden(**f)
        b = csa_golden(**f)
        self.assertEqual(a['output'].shape, (8,4,8))
        self.assertEqual(a['requests'][0]['compressed'].shape, (2,8))
        self.assertEqual(a['requests'][0]['main_state']['positions'], list(range(1,8)))
        self.assertTrue(torch.isfinite(a['output']).all())
        self.assertTrue(torch.equal(a['output'],b['output']))
        self.assertTrue(torch.equal(f['x'],x0))
        self.assertEqual(f['requests'][0]['main_state']['positions'], [])
        self.assertTrue((a['topk'][:3] == -1).all())
        self.assertTrue((a['topk'][3:7,0] == 0).all())

    def test_incremental_cache(self):
        f = fixture()
        whole = csa_golden(**f)
        first = dict(f, x=f['x'][:5], requests=[dict(f['requests'][0], query_length=5)])
        initial = csa_golden(**first)
        second = dict(f, x=f['x'][5:], requests=[dict(initial['requests'][0], query_length=3)])
        tail = csa_golden(**second)
        torch.testing.assert_close(tail['output'],whole['output'][5:],atol=1e-3,rtol=1e-3)
        torch.testing.assert_close(tail['requests'][0]['compressed'],whole['requests'][0]['compressed'])

    def test_reject_lengths(self):
        f = fixture()
        f['requests'][0]['query_length'] = 7
        with self.assertRaises(ValueError):
            csa_golden(**f)


if __name__ == '__main__':
    unittest.main()
