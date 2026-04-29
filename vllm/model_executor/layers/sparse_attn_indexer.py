# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits, has_deep_gemm
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops as ops

logger = init_logger(__name__)


def _dequantize_indexer_k(
    k_fp8: torch.Tensor,
    k_scale_bytes: torch.Tensor,
) -> torch.Tensor:
    scales = k_scale_bytes.contiguous().view(torch.float32).reshape(-1, 1)
    return k_fp8.to(torch.float32) * scales


def _weighted_indexer_logits(
    q_fp8: torch.Tensor,
    k_dequant: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    logits = torch.einsum("qhd,kd->qhk", q_fp8.to(torch.float32), k_dequant)
    return (logits * weights.to(torch.float32).unsqueeze(-1)).sum(dim=1)


def _gather_indexer_k_quant_cache_torch(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    head_dim: int,
    total_seq_lens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = int(seq_lens.shape[0])
    cache_block_size = int(kv_cache.shape[1])
    scale_bytes_per_token = int(kv_cache.shape[2]) - head_dim
    assert scale_bytes_per_token > 0

    device = seq_lens.device
    seq_lens_long = seq_lens.to(torch.long)
    req_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=device, dtype=torch.long),
        seq_lens_long,
        output_size=total_seq_lens,
    )
    token_offsets = torch.arange(
        total_seq_lens,
        device=device,
        dtype=torch.long,
    )
    req_starts = torch.cumsum(seq_lens_long, dim=0) - seq_lens_long
    token_offsets -= req_starts[req_ids]

    block_offsets = torch.div(
        token_offsets,
        cache_block_size,
        rounding_mode="floor",
    )
    valid = block_offsets < block_table.shape[1]
    safe_block_offsets = block_offsets.clamp(max=block_table.shape[1] - 1)
    block_nums = block_table[req_ids, safe_block_offsets].to(torch.long)
    valid &= (block_nums >= 0) & (block_nums < kv_cache.shape[0])
    safe_block_nums = block_nums.clamp(min=0, max=kv_cache.shape[0] - 1)

    page_view = torch.as_strided(
        kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8),
        size=(kv_cache.shape[0], kv_cache.stride(0)),
        stride=(kv_cache.stride(0), 1),
    )
    token_in_block = token_offsets % cache_block_size
    k_offsets = (
        token_in_block[:, None] * head_dim
        + torch.arange(head_dim, device=device, dtype=torch.long)
    )
    scale_offsets = (
        cache_block_size * head_dim
        + token_in_block[:, None] * scale_bytes_per_token
        + torch.arange(scale_bytes_per_token, device=device, dtype=torch.long)
    )

    k_bytes = page_view[safe_block_nums[:, None], k_offsets]
    scale_bytes = page_view[safe_block_nums[:, None], scale_offsets]
    invalid = ~valid
    k_bytes = k_bytes.masked_fill(invalid[:, None], 0)
    scale_bytes = scale_bytes.masked_fill(invalid[:, None], 0)

    return k_bytes.contiguous().view(torch.float8_e4m3fn), scale_bytes.contiguous()


def _topk_per_row_prefill_fallback(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
) -> None:
    topk_indices.fill_(-1)
    num_rows = logits.shape[0]
    for row_idx in range(num_rows):
        row_start = int(cu_seqlen_ks[row_idx].item())
        row_end = int(cu_seqlen_ke[row_idx].item())
        valid_len = max(0, row_end - row_start)
        if valid_len == 0:
            continue
        k = min(topk_tokens, valid_len)
        _, indices = torch.topk(
            logits[row_idx, row_start:row_end],
            k=k,
            dim=-1,
            largest=True,
            sorted=False,
        )
        topk_indices[row_idx, :k] = indices.to(torch.int32)


def _topk_per_row_decode_fallback(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    head_dim: int,
    topk_tokens: int,
) -> None:
    seq_lens = decode_metadata.seq_lens
    seq_lens_cpu = decode_metadata.seq_lens_cpu
    decode_lens_cpu = decode_metadata.decode_lens_cpu
    num_reqs = len(seq_lens_cpu)
    total_seq_lens = decode_metadata.total_seq_lens
    num_decode_tokens = decode_metadata.num_decode_tokens

    if total_seq_lens == 0:
        topk_indices_buffer[:num_decode_tokens].fill_(-1)
        return

    k_fp8_full, k_scale_full = _gather_indexer_k_quant_cache_torch(
        kv_cache,
        decode_metadata.block_table,
        seq_lens,
        head_dim,
        total_seq_lens,
    )
    k_dequant_full = _dequantize_indexer_k(k_fp8_full, k_scale_full)

    topk_indices_buffer[:num_decode_tokens].fill_(-1)
    query_start = 0
    kv_start = 0
    for req_idx in range(num_reqs):
        seq_len = seq_lens_cpu[req_idx]
        decode_len = decode_lens_cpu[req_idx]
        req_k = k_dequant_full[kv_start : kv_start + seq_len]
        kv_start += seq_len

        for local_query_idx in range(decode_len):
            valid_len = seq_len - decode_len + local_query_idx + 1
            valid_len = max(0, min(valid_len, seq_len))
            if valid_len == 0:
                query_start += 1
                continue

            row_logits = _weighted_indexer_logits(
                q_fp8[query_start : query_start + 1],
                req_k[:valid_len],
                weights[query_start : query_start + 1],
            )[0]
            k = min(topk_tokens, valid_len)
            _, indices = torch.topk(
                row_logits,
                k=k,
                dim=-1,
                largest=True,
                sorted=False,
            )
            topk_indices_buffer[query_start, :k] = indices.to(torch.int32)
            query_start += 1


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
            ((total_seq_lens, 4), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    num_tokens = slot_mapping.shape[0]

    if not skip_k_cache_insert:
        # During speculative decoding, k may be padded to the CUDA graph batch
        # size while slot_mapping only covers actual tokens. Truncate k to avoid
        # out-of-bounds reads in the kernel.
        k = k[:num_tokens]
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        for chunk in prefill_metadata.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if current_platform.is_cuda() and not has_deep_gemm():
                logits = _weighted_indexer_logits(
                    q_fp8[chunk.token_start : chunk.token_end],
                    _dequantize_indexer_k(k_fp8, k_scale),
                    weights[chunk.token_start : chunk.token_end],
                )
                _topk_per_row_prefill_fallback(
                    logits,
                    topk_indices,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_tokens,
                )
            else:
                logits = fp8_mqa_logits(
                    q_fp8[chunk.token_start : chunk.token_end],
                    (k_fp8, k_scale.view(torch.float32).flatten()),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
                num_rows = logits.shape[0]
                if current_platform.is_xpu():
                    ops.top_k_per_row_prefill(
                        logits,
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )
                else:
                    torch.ops._C.top_k_per_row_prefill(
                        logits,
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )

            # Compute lengths from row spans
            # lengths = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
            # torch.ops._C.large_context_topk(
            #    logits,
            #    topk_indices,
            #    lengths,
            #    chunk.cu_seqlen_ks,  # row_starts
            # )

    if has_decode:
        decode_metadata = attn_metadata.decode
        assert decode_metadata is not None
        if current_platform.is_cuda() and not has_deep_gemm():
            _topk_per_row_decode_fallback(
                q_fp8[:num_decode_tokens],
                kv_cache,
                weights[:num_decode_tokens],
                decode_metadata,
                topk_indices_buffer,
                head_dim,
                topk_tokens,
            )
        else:
            # kv_cache shape [
            # kv_cache size requirement [num_block, block_size, n_head, head_dim],
            # we only have [num_block, block_size, head_dim],
            kv_cache = kv_cache.unsqueeze(-2)
            decode_lens = decode_metadata.decode_lens
            if decode_metadata.requires_padding:
                # pad in edge case where we have short chunked prefill length <
                # decode_threshold since we unstrictly split
                # prefill and decode by decode_threshold
                # (currently set to 1 + speculative tokens)
                padded_q_fp8_decode_tokens = pack_seq_triton(
                    q_fp8[:num_decode_tokens], decode_lens
                )
            else:
                padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_fp8.shape[1:]
                )
            # TODO: move and optimize below logic with triton kernels
            batch_size = padded_q_fp8_decode_tokens.shape[0]
            next_n = padded_q_fp8_decode_tokens.shape[1]
            assert batch_size == decode_metadata.seq_lens.shape[0]
            num_padded_tokens = batch_size * next_n
            logits = fp8_paged_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
            num_rows = logits.shape[0]
            topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

            if decode_metadata.use_large_context_topk:
                if next_n == 1:
                    lengths = decode_metadata.seq_lens
                else:
                    # (bs,) -> (bs, 1) + (next_n,) -> (bs, next_n) -> (bs * next_n,)
                    lengths = (
                        decode_metadata.seq_lens.unsqueeze(1)
                        - next_n
                        + 1
                        + decode_metadata.offsets
                    ).flatten()

                torch.ops._C.large_context_topk(
                    logits,
                    topk_indices,
                    lengths,
                    None,
                )
            else:
                if current_platform.is_xpu():
                    ops.top_k_per_row_decode(
                        logits,
                        next_n,
                        decode_metadata.seq_lens,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )
                else:
                    torch.ops._C.top_k_per_row_decode(
                        logits,
                        next_n,
                        decode_metadata.seq_lens,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )

            if decode_metadata.requires_padding:
                # if padded, we need to unpack
                # the topk indices removing padded tokens
                topk_indices = unpack_seq_triton(
                    topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                    decode_lens,
                )
                topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                    topk_indices
                )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm() and use_fp4_cache:
            raise RuntimeError(
                "Sparse Attention Indexer CUDA FP4 cache path requires DeepGEMM "
                "to be installed."
            )
        if current_platform.is_cuda() and not has_deep_gemm():
            logger.warning_once(
                "DeepGEMM is unavailable; SparseAttnIndexer will use a slower "
                "PyTorch fallback on CUDA."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache,
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
