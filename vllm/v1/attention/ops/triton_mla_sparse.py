# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA kernels for DeepSeek V4.

These kernels are intentionally simple: they stream sparse indices directly from
the paged KV cache and keep only one query/head accumulator in registers. That
keeps the path under SM120's 48 KB shared-memory ceiling, unlike FlashMLA's
SM90/SM100 sparse kernels.
"""

import torch

from vllm.triton_utils import tl, triton


_DSV4_NOPE_DIM = 448
_DSV4_ROPE_DIM = 64
_DSV4_HEAD_DIM = _DSV4_NOPE_DIM + _DSV4_ROPE_DIM
_DSV4_TOKEN_DATA_SIZE = _DSV4_NOPE_DIM + _DSV4_ROPE_DIM * 2
_DSV4_SCALE_DIM = 8
_DSV4_QUANT_BLOCK = 64

_DSV4_NOPE_DIM_TL = tl.constexpr(448)
_DSV4_HEAD_DIM_TL = tl.constexpr(512)
_DSV4_TOKEN_DATA_SIZE_TL = tl.constexpr(576)
_DSV4_SCALE_DIM_TL = tl.constexpr(8)
_DSV4_QUANT_BLOCK_TL = tl.constexpr(64)


@triton.jit
def _accumulate_dsv4_cache(
    q,
    cache,
    indices,
    lengths,
    token_idx,
    head_idx,
    q_stride0,
    q_stride1,
    cache_stride0,
    idx_stride0,
    cache_block_size: tl.constexpr,
    topk: tl.constexpr,
    start: tl.constexpr,
    block_n: tl.constexpr,
    offs_d: tl.constexpr,
    q_vec,
    sm_scale: tl.constexpr,
    m_i,
    l_i,
    acc,
):
    offs_n = start + tl.arange(0, block_n)
    length = tl.load(lengths + token_idx)
    valid_n = offs_n < length

    slots = tl.load(
        indices + token_idx * idx_stride0 + offs_n,
        mask=(offs_n < topk) & valid_n,
        other=-1,
    )
    valid_n = valid_n & (slots >= 0)

    block_idx = slots // cache_block_size
    pos_in_block = slots - block_idx * cache_block_size
    block_base = cache + block_idx.to(tl.int64) * cache_stride0
    token_base = block_base + pos_in_block * _DSV4_TOKEN_DATA_SIZE_TL
    scale_base = (
        block_base
        + cache_block_size * _DSV4_TOKEN_DATA_SIZE_TL
        + pos_in_block * _DSV4_SCALE_DIM_TL
    )

    is_nope = offs_d < _DSV4_NOPE_DIM_TL
    scale_idx = offs_d // _DSV4_QUANT_BLOCK_TL
    fp8_u8 = tl.load(
        token_base[:, None] + offs_d[None, :],
        mask=valid_n[:, None] & is_nope[None, :],
        other=0,
    )
    fp8_val = fp8_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    encoded_scale = tl.load(
        scale_base[:, None] + scale_idx[None, :],
        mask=valid_n[:, None] & is_nope[None, :],
        other=127,
    )
    scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
    k_nope = fp8_val * scale

    rope_offsets = offs_d - _DSV4_NOPE_DIM_TL
    rope_ptr = (token_base + _DSV4_NOPE_DIM_TL).to(
        tl.pointer_type(tl.bfloat16)
    )
    k_rope = tl.load(
        rope_ptr[:, None] + rope_offsets[None, :],
        mask=valid_n[:, None] & (offs_d[None, :] >= _DSV4_NOPE_DIM_TL),
        other=0.0,
    ).to(tl.float32)
    k = tl.where(is_nope[None, :], k_nope, k_rope)

    qk = tl.sum(q_vec[None, :] * k, axis=1) * sm_scale
    qk = tl.where(valid_n, qk, -float("inf"))

    m_new = tl.maximum(m_i, tl.max(qk, axis=0))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(qk - m_new)
    acc = acc * alpha + tl.sum(p[:, None] * k, axis=0)
    l_i = l_i * alpha + tl.sum(p, axis=0)
    m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _deepseek_v4_sparse_decode_kernel(
    out,
    q,
    swa_cache,
    swa_indices,
    swa_lens,
    attn_sink,
    extra_cache,
    extra_indices,
    extra_lens,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    swa_cache_stride0,
    swa_idx_stride0,
    extra_cache_stride0,
    extra_idx_stride0,
    sm_scale: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    swa_topk: tl.constexpr,
    extra_topk: tl.constexpr,
    has_extra: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs_d = tl.arange(0, head_dim)

    q_vec = tl.load(
        q + token_idx * q_stride0 + head_idx * q_stride1 + offs_d,
        mask=offs_d < _DSV4_HEAD_DIM_TL,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.load(attn_sink + head_idx).to(tl.float32)
    l_i = tl.full((), 1.0, dtype=tl.float32)
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    for start in tl.static_range(0, swa_topk, block_n):
        m_i, l_i, acc = _accumulate_dsv4_cache(
            q,
            swa_cache,
            swa_indices,
            swa_lens,
            token_idx,
            head_idx,
            q_stride0,
            q_stride1,
            swa_cache_stride0,
            swa_idx_stride0,
            swa_block_size,
            swa_topk,
            start,
            block_n,
            offs_d,
            q_vec,
            sm_scale,
            m_i,
            l_i,
            acc,
        )

    if has_extra:
        for start in tl.static_range(0, extra_topk, block_n):
            m_i, l_i, acc = _accumulate_dsv4_cache(
                q,
                extra_cache,
                extra_indices,
                extra_lens,
                token_idx,
                head_idx,
                q_stride0,
                q_stride1,
                extra_cache_stride0,
                extra_idx_stride0,
                extra_block_size,
                extra_topk,
                start,
                block_n,
                offs_d,
                q_vec,
                sm_scale,
                m_i,
                l_i,
                acc,
            )

    result = acc / l_i
    tl.store(
        out + token_idx * out_stride0 + head_idx * out_stride1 + offs_d,
        result,
        mask=offs_d < _DSV4_HEAD_DIM_TL,
    )


@triton.jit
def _sparse_bf16_attention_kernel(
    out,
    q,
    kv,
    indices,
    lengths,
    attn_sink,
    q_stride0,
    q_stride1,
    kv_stride0,
    kv_stride2,
    idx_stride0,
    out_stride0,
    out_stride1,
    sm_scale: tl.constexpr,
    topk: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs_d = tl.arange(0, head_dim)

    q_vec = tl.load(
        q + token_idx * q_stride0 + head_idx * q_stride1 + offs_d,
        mask=offs_d < _DSV4_HEAD_DIM_TL,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.load(attn_sink + head_idx).to(tl.float32)
    l_i = tl.full((), 1.0, dtype=tl.float32)
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    for start in tl.static_range(0, topk, block_n):
        offs_n = start + tl.arange(0, block_n)
        length = tl.load(lengths + token_idx)
        valid_n = offs_n < length
        slots = tl.load(
            indices + token_idx * idx_stride0 + offs_n,
            mask=(offs_n < topk) & valid_n,
            other=-1,
        )
        valid_n = valid_n & (slots >= 0)

        k = tl.load(
            kv + slots[:, None] * kv_stride0 + offs_d[None, :] * kv_stride2,
            mask=valid_n[:, None] & (offs_d[None, :] < _DSV4_HEAD_DIM_TL),
            other=0.0,
        ).to(tl.float32)

        qk = tl.sum(q_vec[None, :] * k, axis=1) * sm_scale
        qk = tl.where(valid_n, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        acc = acc * alpha + tl.sum(p[:, None] * k, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    result = acc / l_i
    tl.store(
        out + token_idx * out_stride0 + head_idx * out_stride1 + offs_d,
        result,
        mask=offs_d < _DSV4_HEAD_DIM_TL,
    )


def _as_2d_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.dim() == 3:
        assert indices.shape[1] == 1
        return indices.squeeze(1)
    assert indices.dim() == 2
    return indices


def triton_deepseek_v4_sparse_decode(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    block_n: int = 16,
) -> None:
    if q.numel() == 0:
        return
    # DeepSeek V4's FlashMLA call path passes decode queries as
    # [tokens, 1, heads, dim].  The Triton fallback works directly on
    # [tokens, heads, dim].
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q.squeeze(1)
    assert q.shape[-1] == _DSV4_HEAD_DIM
    assert out.shape[0] == q.shape[0]
    assert out.shape[-1] == q.shape[-1]
    num_heads = min(q.shape[1], out.shape[1])
    q = q[:, :num_heads, :]
    out = out[:, :num_heads, :]
    swa_indices = _as_2d_indices(swa_indices)

    has_extra = extra_cache is not None and extra_indices is not None
    if has_extra:
        assert extra_lens is not None
        extra_indices = _as_2d_indices(extra_indices)
        extra_topk = extra_indices.shape[-1]
        extra_cache_stride0 = extra_cache.stride(0)
        extra_idx_stride0 = extra_indices.stride(0)
        extra_block_size = extra_cache.shape[1]
    else:
        assert extra_lens is None
        extra_cache = swa_cache
        extra_indices = swa_indices
        extra_lens = swa_lens
        extra_topk = 1
        extra_cache_stride0 = swa_cache.stride(0)
        extra_idx_stride0 = swa_indices.stride(0)
        extra_block_size = swa_cache.shape[1]

    grid = (q.shape[0], q.shape[1])
    _deepseek_v4_sparse_decode_kernel[grid](
        out,
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        attn_sink,
        extra_cache,
        extra_indices,
        extra_lens,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        swa_cache.stride(0),
        swa_indices.stride(0),
        extra_cache_stride0,
        extra_idx_stride0,
        sm_scale,
        swa_cache.shape[1],
        extra_block_size,
        swa_indices.shape[-1],
        extra_topk,
        has_extra,
        block_n,
        triton.next_power_of_2(q.shape[-1]),
    )


def triton_sparse_bf16_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor,
    block_n: int = 16,
) -> None:
    if q.numel() == 0:
        return
    assert q.shape[-1] == _DSV4_HEAD_DIM
    assert out.shape[0] == q.shape[0]
    assert out.shape[-1] == q.shape[-1]
    num_heads = min(q.shape[1], out.shape[1])
    q = q[:, :num_heads, :]
    out = out[:, :num_heads, :]
    indices = _as_2d_indices(indices)
    assert kv.dim() == 3 and kv.shape[1] == 1

    grid = (q.shape[0], q.shape[1])
    _sparse_bf16_attention_kernel[grid](
        out,
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        kv.stride(2),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        sm_scale,
        indices.shape[-1],
        block_n,
        triton.next_power_of_2(q.shape[-1]),
    )
