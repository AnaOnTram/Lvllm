# Continue - DeepSeek V4 Flash / Blackwell FlashMLA

## STATUS: SM120 Hardware Blocker — Requires New Kernel Work

DeepSeek V4-Flash cannot serve on RTX PRO 2000 (SM120) with the current
implementation. This is a hardware limitation, not a configuration issue.

---

## Root Cause

`vllm/model_executor/layers/deepseek_v4_attention.py:651-657` has two hard assertions:

```python
assert kv_cache_dtype.startswith("fp8")
assert issubclass(self.get_attn_backend(), FlashMLASparseBackend)
```

DeepSeek V4 requires:
1. `fp8_ds_mla` KV cache format (656 bytes/token, includes scales + RoPE part)
2. FlashMLA Sparse backend for SM90/SM100

SM120 (RTX PRO 2000 Blackwell GB202) cannot satisfy either requirement:
- `max_smem_optin per block = 0 bytes` — no opt-in shared memory beyond 48 KB
- FlashMLA SM90 and SM100 kernels both call
  `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedMemoryPlan))`
  which fails with `cudaErrorInvalidValue` because SharedMemoryPlan >> 48 KB

---

## Full Failure Progression (complete picture)

| Step | Failure | Fix Applied |
|------|---------|-------------|
| 1 | `moe_wna16_marlin_gemm` missing | Rebuilt `_moe_C` |
| 2 | KV cache page size not divisible | `kv_cache_utils.py` heterogeneous fallback |
| 3 | `kv_lora_rank` missing from V4 config | `mla_attention.py` field fallback |
| 4 | `_flashmla_C` not compiled | Rebuilt with SM12x arch targets |
| 5 | `Unsupported architecture for sparse decode fwd` | Patched `is_sm100f()` |
| 6 | SM100 kernel `cudaErrorInvalidValue` | Routed SM120 → SM90 kernel path |
| 7 | SM90 kernel `cudaErrorInvalidValue` | SAME HW LIMIT: 48KB smem cap |
| 8 | `DeepseekV4 only supports fp8 kv-cache … got auto` | bf16 cache rejected by model |
| **HW LIMIT** | SM120 has 0 KB opt-in shared memory | **Cannot be patched** |

---

## What Would Be Needed to Unblock SM120

**Option A — Triton sparse fp8 decode kernel (most practical)**
- Write a new Triton kernel that:
  - Reads the `fp8_ds_mla` 656-byte cache format
  - Does TopK block selection (sparse=512)
  - Runs within 48 KB shared memory
- Register it as a new `TRITON_MLA_SPARSE` backend that:
  - Sets `supports_compute_capability → True`
  - Has `is_sparse() → True`
  - Accepts `fp8_ds_mla` in `supported_kv_cache_dtypes`
- Relax the `assert issubclass(self.get_attn_backend(), FlashMLASparseBackend)` to
  also accept the new backend class

**Option B — Dense bf16 fallback (lower effort, less accurate)**
- Relax both assertions in `deepseek_v4_attention.py` (or guard with platform check)
- Allow TRITON_MLA (bf16, dense) as a fallback
- Model will produce correct outputs but attending all blocks (not just TopK=512),
  ~20x more attention compute for max_model_len=10000

---

## Current Remote State

- `_moe_C.abi3.so` — rebuilt, all ops present including `moe_wna16_marlin_gemm`
- `_flashmla_C.abi3.so` — rebuilt with SM120 targets (21:50 timestamp)
- `common.h` patch: `is_sm90f()` accepts `major == 9 || major == 12` (does not help)
- `deepseek_v4.yaml` — restored to `kv_cache_dtype: "fp8"` (serves no function until
  the kernel work is done)

---

## Files Touched (all committed locally on main)

- `cmake/external_projects/flashmla.cmake`
- `vllm/v1/attention/backends/mla/flashmla_sparse.py` (SM120 now excluded)
- `vllm/v1/attention/ops/flashmla.py`
- `vllm/platforms/cuda.py`
- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/core/kv_cache_utils.py`
- `csrc/moe/moe_ops.h`
- `vllm/_custom_ops.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`
- `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`
- `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py`
- `vllm/model_executor/models/deepseek_v4.py`
- `vllm/utils/deep_gemm.py`
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`

---

## Do Not

- Do not attempt more FlashMLA SM90/SM100 kernel routing — both fail with same hw limit.
- Do not revert the `_moe_C` build or MoE fallbacks; they are correct and needed.
- Do not put the SSH password in notes or commits.
