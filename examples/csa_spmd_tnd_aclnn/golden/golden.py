"""Independent, logical-ND CSA mathematical reference (not a PTO emulator).

No vLLM/PyPTO/custom fused operators. Basic torch ops run on the input device.
The reference is intentionally slow, serial per request/token, and not timed.
Physical paged-cache import/export and production tolerance qualification are
separate work: do not claim these outputs are a qualified PTO oracle yet.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    heads: int = 64
    index_heads: int = 64
    head_dim: int = 512
    index_dim: int = 128
    rope_dim: int = 64
    groups: int = 8
    window: int = 128
    topk: int = 512
    ratio: int = 4
    eps: float = 1e-6
    sinkhorn_iters: int = 20


def rms(x, gamma=None, eps=1e-6):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return y if gamma is None else y * gamma.float()


def rope(x, cos, sin, inverse=False):
    """Interleaved RoPE on the final R columns; cos/sin contain R columns."""
    r = cos.shape[-1]
    if r % 2 or r > x.shape[-1]:
        raise ValueError('invalid RoPE dimension')
    tail = x.float()[..., -r:]
    paired = tail.reshape(*tail.shape[:-1], r // 2, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), -1).flatten(-2)
    return torch.cat((x.float()[..., :-r], tail * cos + rotated * (-sin if inverse else sin)), -1)


def quantize(x):
    """Symmetric per-last-axis INT8; zero rows remain zero, no NaN scale."""
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
    return torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8), scale


def int8_mm(a, b):
    """Basic-op integer dot: small exact FP32 partials accumulated in INT32.

    At K<=256, 256*128*128 is representable exactly in FP32. Disable reduced
    precision matmul on the target backend; this is tested on CPU separately.
    """
    if a.dtype != torch.int8 or b.dtype != torch.int8 or a.shape[-1] != b.shape[0]:
        raise ValueError('int8_mm expects INT8 [..., K] and [K, N]')
    result = torch.zeros((*a.shape[:-1], b.shape[1]), dtype=torch.int32, device=a.device)
    for k in range(0, b.shape[0], 256):
        partial = a[..., k:k + 256].float() @ b[k:k + 256].float()
        result = result + partial.to(torch.int32)
    return result


def quant_linear(q, activation_scale, weight, weight_scale):
    return (int8_mm(q, weight).float() * activation_scale) * weight_scale.float()


def hc_pre(x, weight, scale, base, cfg):
    width = x.shape[-2]
    flat = x.float().flatten(-2)
    mixes = F.linear(flat, weight.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.eps)
    pre, post, comb = mixes.split((width, width, width * width), -1)
    pre = torch.sigmoid(pre * scale[0] + base[:width]) + cfg.eps
    post = 2 * torch.sigmoid(post * scale[1] + base[width:2 * width])
    comb = (comb * scale[2] + base[2 * width:]).reshape(-1, width, width)
    comb = torch.softmax(comb, -1) + cfg.eps
    comb = comb / (comb.sum(-2, keepdim=True) + cfg.eps)
    for _ in range(cfg.sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + cfg.eps)
        comb = comb / (comb.sum(-2, keepdim=True) + cfg.eps)
    mixed = (pre.unsqueeze(-1) * x.float()).sum(-2).to(torch.bfloat16)
    return mixed, post, comb


def hc_post(attention, residual, post, comb):
    # comb[input HC lane, output HC lane], not the transpose.
    y = attention.float().unsqueeze(-2) * post.unsqueeze(-1)
    for lane in range(residual.shape[-2]):
        y = y + residual[:, lane, :].float().unsqueeze(1) * comb[:, lane, :].unsqueeze(-1)
    return y.to(torch.bfloat16)


def compressor_step(x, position, state, weight, gate, ape, norm, cos, sin, cfg):
    """State holds last <=7 logical positions, values and APE-added logits.

    Overlap window: older four use first projection half, newer four second.
    Missing negative positions have zero probability. Does not mutate inputs.
    """
    values = torch.cat((state['values'], F.linear(x.float(), weight.float()).unsqueeze(0)))
    logits = torch.cat((state['logits'], (F.linear(x.float(), gate.float()) + ape[position % cfg.ratio]).unsqueeze(0)))
    positions = state['positions'] + [position]
    pooled = None
    if (position + 1) % cfg.ratio == 0:
        rows, scores = [], []
        dim = norm.numel()
        for p in range(max(0, position - 2 * cfg.ratio + 1), position + 1):
            if p not in positions:
                raise ValueError(f'missing compressor history at position {p}')
            i = positions.index(p)
            half = 0 if p <= position - cfg.ratio else dim
            rows.append(values[i, half:half + dim])
            scores.append(logits[i, half:half + dim])
        pooled = (torch.softmax(torch.stack(scores), dim=0) * torch.stack(rows)).sum(0)
        pooled = rope(rms(pooled, norm, cfg.eps), cos, sin).to(torch.bfloat16)
    keep = 2 * cfg.ratio - 1
    return pooled, {'values': values[-keep:].clone(), 'logits': logits[-keep:].clone(), 'positions': positions[-keep:]}


@torch.inference_mode()
def csa_golden(x, weights, requests, rope_cos, rope_sin, cfg=Config()):
    """Run logical TND requests; Q is expressed as request query_lengths.

    x: active tokens only [T,HC,D], BF16. weights: logical ND tensors (see
    README). requests: list of histories/caches, one per request. rope tables:
    [max_absolute_position+1,R], FP32, same YaRN frequencies as tested op.
    Returns output, topk/scores, intermediates and independent updated caches.
    No tolerance decision or comparison with the candidate is made here.
    """
    if x.dtype != torch.bfloat16 or x.ndim != 3:
        raise ValueError('x must be BF16 [T,HC,D]')
    if cfg.ratio != 4 or cfg.rope_dim > min(cfg.head_dim, cfg.index_dim):
        raise ValueError('only ratio4 with valid RoPE dimensions is supported')
    if sum(r['query_length'] for r in requests) != x.shape[0]:
        raise ValueError('query lengths must cover exactly the active tokens')
    if any(r['query_length'] < 1 or r['history'] < 0 for r in requests):
        raise ValueError('invalid request lengths')
    w = weights
    mixed, post, comb = hc_pre(x, w['hc_attn_fn'], w['hc_attn_scale'], w['hc_attn_base'], cfg)
    normed = rms(mixed, w['attn_norm_w'], cfg.eps).to(torch.bfloat16)
    qr = F.linear(normed.float(), w['wq_a'].float()).to(torch.bfloat16)
    # Native fused RMSNormDynamicQuant quantizes the FP32 normalized value.
    qr = rms(qr, w['gamma_cq'], cfg.eps)
    qr_q, qr_scale = quantize(qr)
    query = quant_linear(qr_q, qr_scale, w['wq_b'], w['wq_b_scale']).reshape(-1, cfg.heads, cfg.head_dim)
    query = rms(query.to(torch.bfloat16), eps=cfg.eps).to(torch.bfloat16)
    kv = rms((normed.float() @ w['wkv'].float()).to(torch.bfloat16), w['gamma_ckv'], cfg.eps).to(torch.bfloat16)
    iq = quant_linear(qr_q, qr_scale, w['idx_wq_b'], w['idx_wq_b_scale']).to(torch.bfloat16).reshape(-1, cfg.index_heads, cfg.index_dim)
    head_weights = (normed.float() @ w['weights_proj'].float()).to(torch.bfloat16)
    head_weights = (head_weights.float() * (cfg.index_dim * cfg.index_heads) ** -0.5).to(torch.bfloat16).half().float()
    attended, all_topk, all_scores, updated = [], [], [], []
    offset = 0
    for request in requests:
        h, count = request['history'], request['query_length']
        # Cache rows are dense logical positions, never physical pages/NZ.
        swa = request['swa'].clone()
        compressed = request['compressed'].clone()
        keys = request['index_keys'].clone()
        scales = request['index_scales'].clone()
        if swa.shape != (h, cfg.head_dim) or compressed.shape != (h // 4, cfg.head_dim):
            raise ValueError('logical attention cache shape/history mismatch')
        if keys.shape != (h // 4, cfg.index_dim) or scales.shape != (h // 4, 1):
            raise ValueError('logical index cache shape/history mismatch')
        main_state, idx_state = request['main_state'], request['index_state']
        for j in range(count):
            t, position = offset + j, h + j
            cos, sin = rope_cos[position], rope_sin[position]
            q = rope(query[t], cos, sin).to(torch.bfloat16)
            new_kv = rope(kv[t], cos, sin).to(torch.bfloat16)
            swa = torch.cat((swa, new_kv[None]))
            cc, ss = rope_cos[(position // 4) * 4], rope_sin[(position // 4) * 4]
            pooled, main_state = compressor_step(normed[t], position, main_state, w['cmp_wkv'], w['cmp_wgate'], w['cmp_ape'], w['cmp_norm_w'], cc, ss, cfg)
            ik, idx_state = compressor_step(normed[t], position, idx_state, w['inner_wkv'], w['inner_wgate'], w['inner_ape'], w['inner_norm_w'], cc, ss, cfg)
            if pooled is not None:
                compressed = torch.cat((compressed, pooled[None]))
                rotated_key = F.linear(ik.float(), w['hadamard_idx'].float()).to(torch.bfloat16)
                rotated_key = (rotated_key.float() * cfg.index_dim ** -0.5).to(torch.bfloat16)
                key_q, key_s = quantize(rotated_key)
                keys = torch.cat((keys, key_q[None]))
                scales = torch.cat((scales, key_s.half().float().reshape(1, 1)))
            rotated_q = rope(iq[t], cos, sin).to(torch.bfloat16)
            rotated_q = F.linear(rotated_q.float(), w['hadamard_idx'].float()).to(torch.bfloat16)
            rotated_q = (rotated_q.float() * cfg.index_dim ** -0.5).to(torch.bfloat16)
            index_q, index_qscale = quantize(rotated_q)
            index_qscale = index_qscale.half().float()
            scores = ((int8_mm(index_q, keys.T).float() * index_qscale) * scales.T).relu()
            scores = (scores * head_weights[t, :, None]).sum(0)
            # Stable descending score, ascending logical index for ties.
            chosen = torch.argsort(scores, descending=True, stable=True)[:cfg.topk]
            top = torch.full((cfg.topk,), -1, dtype=torch.int32, device=x.device)
            top_scores = torch.zeros(cfg.topk, device=x.device)
            top[:chosen.numel()] = chosen.to(torch.int32)
            top_scores[:chosen.numel()] = scores[chosen]
            all_topk.append(top)
            all_scores.append(top_scores)
            selected = torch.cat((swa[max(0, position + 1 - cfg.window):], compressed[chosen]))
            logits = q.float() @ selected.float().T * cfg.head_dim ** -0.5
            logits = torch.cat((logits, w['attn_sink'].float().reshape(cfg.heads, 1)), -1)
            probability = torch.softmax(logits, -1)[:, :-1]
            value = probability @ selected.float()
            attended.append(rope(value, cos, sin, inverse=True).to(torch.bfloat16))
        updated.append({'history': h + count, 'swa': swa, 'compressed': compressed, 'index_keys': keys,
                        'index_scales': scales, 'main_state': main_state, 'index_state': idx_state})
        offset += count
    attention = torch.stack(attended)
    grouped = attention.reshape(x.shape[0], cfg.groups, -1)
    oa = torch.stack([grouped[:, g].float() @ w['wo_a'][g].float() for g in range(cfg.groups)], 1)
    # Native dsa_v1: BF16 batch-matmul -> flatten all groups -> one
    # dynamic scale per token for wo_b, not PTO's per-group scale.
    oa_flat = oa.to(torch.bfloat16).flatten(1)
    oq, oscale = quantize(oa_flat)
    projected = quant_linear(oq, oscale, w['wo_b'], w['wo_b_scale']).to(torch.bfloat16)
    return {'output': hc_post(projected, x, post, comb), 'topk': torch.stack(all_topk),
            'scores': torch.stack(all_scores), 'requests': updated,
            'intermediates': {'hc_mixed': mixed, 'hc_post': post, 'hc_comb': comb,
                              'normed': normed, 'qr': qr, 'attention': attention, 'projected': projected}}
