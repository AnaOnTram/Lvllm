# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

import functools
import importlib
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, NoReturn

import torch

import vllm.envs as envs
from vllm.logger import logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv

_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES: set[str] = {
    "qwen3_5_text",
    "qwen3_5_moe_text",
}


def should_auto_disable_deep_gemm(model_type: str | None) -> bool:
    """Check if DeepGemm should be auto-disabled for this model on Blackwell.

    Returns True if the model is known to have accuracy degradation with
    DeepGemm's E8M0 scale format on Blackwell GPUs (SM100+).
    """
    if model_type is None:
        return False
    if not current_platform.is_device_capability_family(100):
        return False
    return model_type in _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES


class DeepGemmQuantScaleFMT(Enum):
    # Float32 scales in Float32 tensor
    FLOAT32 = 0
    # Compute float32 scales and ceil the scales to UE8M0.
    # Keep the scales in Float32 tensor.
    FLOAT32_CEIL_UE8M0 = 1
    # Compute float32 scales and ceil the scales to UE8M0.
    # Pack the scales into a int32 tensor where each int32
    # element contains 4 scale values.
    UE8M0 = 2

    @classmethod
    def init_oracle_cache(cls) -> None:
        """Initialize the oracle decision and store it in the class cache"""
        cached = getattr(cls, "_oracle_cache", None)
        if cached is not None:
            return

        use_e8m0 = (
            envs.VLLM_USE_DEEP_GEMM_E8M0
            and is_deep_gemm_supported()
            and (_fp8_gemm_nt_impl is not None)
        )
        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return

        cls._oracle_cache = (  # type: ignore
            cls.UE8M0
            if current_platform.is_device_capability_family(100)
            else cls.FLOAT32_CEIL_UE8M0
        )

    @classmethod
    def from_oracle(cls) -> "DeepGemmQuantScaleFMT":
        """Return the pre-initialized oracle decision"""
        cached = getattr(cls, "_oracle_cache", None)
        assert cached is not None, "DeepGemmQuantScaleFMT oracle cache not initialized"
        return cached


@functools.cache
def is_deep_gemm_supported() -> bool:
    """Return `True` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    is_supported_arch = current_platform.support_deep_gemm()
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    """Return `True` if vLLM is configured to use DeepGEMM "
    "E8M0 scale on a Hopper or Blackwell-class GPU.
    """
    if not is_deep_gemm_supported():
        logger.debug_once(
            "DeepGEMM E8M0 disabled: DeepGEMM not supported on this system."
        )
        return False

    _lazy_init()

    if _fp8_gemm_nt_impl is None:
        logger.info_once(
            "DeepGEMM E8M0 disabled: _fp8_gemm_nt_impl not found", scope="local"
        )
        return False

    if envs.VLLM_USE_DEEP_GEMM_E8M0:
        logger.info_once("DeepGEMM E8M0 enabled on current platform.", scope="local")
        return True

    logger.info_once("DeepGEMM E8M0 disabled on current configuration.", scope="local")
    return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is not available or outdated. Please install or "
        "update the `deep_gemm` to a newer version to enable FP8 kernels."
    )


_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_einsum_impl: Callable[..., Any] | None = None
_grouped_impl: Callable[..., Any] | None = None
_grouped_masked_impl: Callable[..., Any] | None = None
_grouped_fp4_impl: Callable[..., Any] | None = None
_fp8_mqa_logits_impl: Callable[..., Any] | None = None
_fp8_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_paged_mqa_logits_metadata_impl: Callable[..., Any] | None = None
_get_mn_major_tma_aligned_tensor_impl: Callable[..., Any] | None = None
_get_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = None
_transform_sf_into_required_layout_impl: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global \
        _fp8_gemm_nt_impl, \
        _fp8_einsum_impl, \
        _grouped_impl, \
        _grouped_masked_impl, \
        _grouped_fp4_impl
    global _fp8_mqa_logits_impl, _fp8_paged_mqa_logits_impl
    global _get_paged_mqa_logits_metadata_impl
    global _get_mn_major_tma_aligned_tensor_impl
    global _get_mk_alignment_for_contiguous_layout_impl
    global _transform_sf_into_required_layout_impl
    global _tf32_hc_prenorm_gemm_impl
    # fast path
    if (
        _fp8_gemm_nt_impl is not None
        or _fp8_einsum_impl is not None
        or _grouped_impl is not None
        or _grouped_masked_impl is not None
        or _grouped_fp4_impl is not None
        or _fp8_mqa_logits_impl is not None
        or _fp8_paged_mqa_logits_impl is not None
        or _get_paged_mqa_logits_metadata_impl is not None
        or _get_mk_alignment_for_contiguous_layout_impl is not None
        or _transform_sf_into_required_layout_impl is not None
        or _tf32_hc_prenorm_gemm_impl is not None
    ):
        return

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm"
        )

    _dg = importlib.import_module("deep_gemm")

    _fp8_gemm_nt_impl = getattr(_dg, "fp8_gemm_nt", None)
    _fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)
    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)
    _grouped_masked_impl = getattr(_dg, "fp8_m_grouped_gemm_nt_masked", None)
    _grouped_fp4_impl = getattr(_dg, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    _fp8_mqa_logits_impl = getattr(_dg, "fp8_mqa_logits", None)
    _fp8_paged_mqa_logits_impl = getattr(_dg, "fp8_paged_mqa_logits", None)
    _get_paged_mqa_logits_metadata_impl = getattr(
        _dg, "get_paged_mqa_logits_metadata", None
    )
    _get_mn_major_tma_aligned_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_tensor", None
    )
    _get_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_mk_alignment_for_contiguous_layout", None
    )
    _transform_sf_into_required_layout_impl = getattr(
        _dg, "transform_sf_into_required_layout", None
    )
    _tf32_hc_prenorm_gemm_impl = getattr(_dg, "tf32_hc_prenorm_gemm", None)
    DeepGemmQuantScaleFMT.init_oracle_cache()


def get_num_sms() -> int:
    _lazy_init()
    _dg = importlib.import_module("deep_gemm")
    return int(_dg.get_num_sms())


@functools.cache
def get_mk_alignment_for_contiguous_layout() -> list[int]:
    _lazy_init()
    if _get_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()
    return [mk_align_size, mk_align_size]


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for DeepGEMM's get_mn_major_tma_aligned_tensor"""
    _lazy_init()
    if _get_mn_major_tma_aligned_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_tensor_impl(x)


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    if "is_deep_gemm_e8m0_used" in kwargs:
        use_ue8m0 = kwargs["is_deep_gemm_e8m0_used"]
        del kwargs["is_deep_gemm_e8m0_used"]
    else:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    return _fp8_gemm_nt_impl(*args, disable_ue8m0_cast=not use_ue8m0, **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def _fp8_einsum_pytorch_fallback(
    equation: str,
    a_pair: tuple[torch.Tensor, torch.Tensor],
    b_pair: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    recipe: tuple[int, ...] | None = None,
) -> None:
    """Pure-PyTorch fallback for fp8_einsum when DeepGEMM is unavailable.

    Handles both FP32 block scales (SM90 / Hopper) and packed UE8M0 INT32
    scales (SM100 / Blackwell).  For UE8M0: each int32 holds 4 uint8 bytes
    where byte i (little-endian) encodes scale_i = 2^(byte_i - 127).
    """

    def _dequantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        t_f32 = tensor.to(torch.float32)

        if scale.dtype == torch.int32:
            # Packed UE8M0 (SM100): 4 UE8M0 bytes per int32, little-endian.
            # byte e  →  float 2^(e - 127)
            scale_u8 = scale.contiguous().view(torch.uint8)
            scale_f32 = torch.exp2(scale_u8.to(torch.float32) - 127.0)
        else:
            scale_f32 = scale.to(torch.float32)

        # Reshape scale dims to match tensor if they differ (e.g. 2-D scale
        # against a 3-D tensor when the weight uses a flat G*L layout).
        if scale_f32.dim() != t_f32.dim():
            scale_f32 = scale_f32.reshape(*t_f32.shape[: t_f32.dim() - 1], -1)

        # Expand blocks in the last (contraction) dimension.
        block_size = t_f32.shape[-1] // scale_f32.shape[-1]
        if block_size > 1:
            scale_f32 = scale_f32.repeat_interleave(block_size, dim=-1)

        return t_f32 * scale_f32

    a, a_scale = a_pair
    b, b_scale = b_pair
    a_dq = _dequantize(a, a_scale)
    b_dq = _dequantize(b, b_scale)
    out.copy_(torch.einsum(equation, a_dq, b_dq).to(out.dtype))


def fp8_einsum(*args, **kwargs):
    _lazy_init()
    if _fp8_einsum_impl is None:
        # DeepGEMM not available — fall back to a pure-PyTorch implementation.
        equation, a_pair, b_pair, out = args[0], args[1], args[2], args[3]
        recipe = kwargs.get("recipe", None)
        return _fp8_einsum_pytorch_fallback(equation, a_pair, b_pair, out, recipe)
    return _fp8_einsum_impl(*args, **kwargs)


def m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_fp4_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_fp4_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """Run the DeepGEMM TF32 pre-norm helper used by the V4 MHC path."""
    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        # PyTorch fallback for GPUs without deep_gemm support.
        # fn: [hc_mult3, hc_hidden_size]; x: [T, hc_hidden_size]
        # out: [n_splits, T, hc_mult3]; sqrsum: [n_splits, T]
        x_f32 = x.to(torch.float32)
        out.zero_()
        sqrsum.zero_()
        out[0] = x_f32 @ fn.T
        sqrsum[0] = (x_f32 * x_f32).sum(dim=-1)
        return
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_masked_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def transform_sf_into_required_layout(*args, **kwargs):
    _lazy_init()
    if _transform_sf_into_required_layout_impl is None:
        return _missing(*args, **kwargs)
    return _transform_sf_into_required_layout_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N])
            with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    _lazy_init()
    if _fp8_mqa_logits_impl is None:
        return _missing()
    return _fp8_mqa_logits_impl(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
    )


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor:
    """Build scheduling metadata for paged MQA logits.

    Args:
        context_lens: Tensor of shape [B], dtype int32; effective context length
            per batch element.
        block_size: KV-cache block size in tokens (e.g., 64).
        num_sms: Number of SMs available. 132 for Hopper

    Returns:
        Backend-specific tensor consumed by `fp8_paged_mqa_logits` to
        schedule work across SMs.
    """
    _lazy_init()
    if _get_paged_mqa_logits_metadata_impl is None:
        return _missing()
    return _get_paged_mqa_logits_metadata_impl(context_lens, block_size, num_sms)


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    _lazy_init()
    if _fp8_paged_mqa_logits_impl is None:
        return _missing()
    return _fp8_paged_mqa_logits_impl(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _align(x: int, y: int) -> int:
    return cdiv(x, y) * y


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/csrc/utils/math.hpp#L19
def get_tma_aligned_size(x: int, element_size: int) -> int:
    return _align(x, 16 // element_size)


DEFAULT_BLOCK_SIZE = [128, 128]


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/dd6ed14acbc7445dcef224248a77ab4d22b5f240/deep_gemm/utils/math.py#L38
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def per_block_cast_to_fp8(
    x: torch.Tensor, block_size: list[int] = DEFAULT_BLOCK_SIZE, use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size
    x_padded = torch.zeros(
        (_align(m, block_m), _align(n, block_n)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    _, fp8_max = get_fp8_min_max()
    sf = x_amax / fp8_max
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(fp8_dtype)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing `torch.testing.assert_close` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report `1 - sim`.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight: torch.Tensor,
    supports_deep_gemm: bool | None = None,
):
    if supports_deep_gemm is None:
        supports_deep_gemm = is_deep_gemm_supported()

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    return (
        supports_deep_gemm
        and output_dtype == torch.bfloat16
        and weight.shape[0] % N_MULTIPLE == 0
        and weight.shape[1] % K_MULTIPLE == 0
    )


__all__ = [
    "calc_diff",
    "DeepGemmQuantScaleFMT",
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "fp8_mqa_logits",
    "fp8_paged_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "per_block_cast_to_fp8",
    "is_deep_gemm_e8m0_used",
    "is_deep_gemm_supported",
    "get_num_sms",
    "should_use_deepgemm_for_fp8_linear",
    "get_col_major_tma_aligned_tensor",
    "get_mk_alignment_for_contiguous_layout",
    "fp8_einsum",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "tf32_hc_prenorm_gemm",
]
