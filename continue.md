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

## 2026-04-28 Local Follow-Up: Option A Started

Local implementation work has started on the Triton sparse path:

- Added `TRITON_MLA_SPARSE` backend registration.
- Added an SM120 backend class in
  `vllm/v1/attention/backends/mla/triton_mla_sparse.py`.
- Added SM120 routing in `vllm/platforms/cuda.py` for quantized sparse MLA.
- Added Triton kernels in `vllm/v1/attention/ops/triton_mla_sparse.py`:
  - sparse decode over DeepSeek V4 fp8_ds_mla cache pages
  - sparse bf16 prefill over the gathered workspace
- Switched `DeepseekV4MLAAttention` to choose the Triton sparse backend on
  compute capability major 12.
- Replaced DeepSeek V4 FlashMLA decode/prefill calls with the Triton kernels
  when the Triton sparse backend is active.

Important detail: DeepSeek V4's actual main/SWA cache layout in this branch is
584 bytes/token at the allocation level, with 576 bytes of token data
(`448 fp8 NoPE + 64 bf16 RoPE`) plus 8 UE8M0 scale bytes stored in the per-block
scale area. This differs from older FlashMLA's 656-byte MLA cache description.

Local verification completed:

```bash
.venv/bin/python -m py_compile \
  vllm/v1/attention/ops/triton_mla_sparse.py \
  vllm/v1/attention/backends/mla/triton_mla_sparse.py \
  vllm/model_executor/layers/deepseek_v4_attention.py \
  vllm/v1/attention/backends/registry.py \
  vllm/platforms/cuda.py

git diff --check -- \
  vllm/v1/attention/ops/triton_mla_sparse.py \
  vllm/v1/attention/backends/mla/triton_mla_sparse.py \
  vllm/model_executor/layers/deepseek_v4_attention.py \
  vllm/v1/attention/backends/registry.py \
  vllm/platforms/cuda.py
```

Both commands passed.

2026-04-29 remote smoke after capture-sync patch:

- Synced and verified:
  - `vllm/model_executor/layers/sparse_attn_indexer.py`
    `0382716a90c558176aa8ac4b57885d07d8d705a4eeef97a599d1ee16d7dae70b`
  - `vllm/v1/attention/backends/mla/indexer.py`
    `a1814b93364531351b420c2d33fe493132a9fce6f2a4c50a0a9a77f1021b6310`
- Rerun with original `gpu_memory_utilization=0.65` plus all code patches:
  - Passed C128A metadata failure.
  - Passed sparse-indexer CUDA graph capture-sync failure.
  - Failed at KV-cache sizing:
    `Available KV cache memory: -3.74 GiB`.
- Rerun with `--gpu-memory-utilization 0.70`:
  - Same KV sizing failure:
    `Available KV cache memory: -2.96 GiB`.
- Rerun with `--gpu-memory-utilization 0.95`:
  - Code path still clean through CUDA graph profiling.
  - KV memory becomes positive but still too small:
    `Available KV cache memory: 0.91 GiB`.
  - vLLM estimates 10k context needs `2.48 GiB` KV cache and the current
    memory budget supports max model length around `2944`.

Current status:

- The original runtime bugs have moved forward substantially:
  - SM120 FlashMLA routing issue is bypassed by Triton sparse MLA.
  - Indexer custom-gather illegal access is bypassed by PyTorch gather fallback.
  - C128A sparse metadata is populated.
  - PyTorch indexer fallback no longer breaks CUDA graph capture via `.item()`.
- The remaining blocker in the latest smoke test is capacity/configuration, not
  the previous attention/indexer exception path.

Likely next runs:

```bash
# Validate server startup with reduced context:
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=20 \
  OMP_NUM_THREADS=20 vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --config deepseek_v4.yaml --cpu-offload-gb 138 \
  --gpu-memory-utilization 0.95 --max-model-len 2900

# If 10k context is mandatory, the next code investigation should target the
# heterogeneous KV cache padding/waste, which reports up to 90.48% waste for
# several groups.
```

2026-04-29 follow-up:

- User ran the 2900-token command and got past graph capture, then hit:

```text
KeyError: <class 'vllm.v1.kv_cache_interface.SlidingWindowMLASpec'>
```

- Root cause: `single_type_kv_cache_manager.py` dispatches managers by exact
  `type(kv_cache_spec)`. It registered `SlidingWindowSpec` but not the
  `SlidingWindowMLASpec` subclass. `SlidingWindowManager` already accepts
  `isinstance(kv_cache_spec, SlidingWindowSpec)`, so the narrow fix is to add
  `SlidingWindowMLASpec: SlidingWindowManager` to `spec_manager_map`.
- Patched and synced:
  - `vllm/v1/core/single_type_kv_cache_manager.py`
  - remote/local hash:
    `17df2353871ca9fdf5bd77abfaa36d907f9cf104a35035c9aca28f0bde66cea9`
- Local verification:

```bash
.venv/bin/python -m py_compile vllm/v1/core/single_type_kv_cache_manager.py
git diff --check -- vllm/v1/core/single_type_kv_cache_manager.py
```

- Remote smoke command:

```bash
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=20 \
  OMP_NUM_THREADS=20 vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --config deepseek_v4.yaml --cpu-offload-gb 138 \
  --gpu-memory-utilization 0.95 --max-model-len 2900 \
  --max-num-batched-tokens 5800
```

- Result: server started successfully and stayed up for 60 seconds before the
  smoke run was manually interrupted to free the port.
- Important successful log lines:

```text
Available KV cache memory: 1.63 GiB
GPU KV cache size: 128 tokens
Maximum concurrency for 2,900 tokens per request: 1.82x
Graph capturing finished in 35 secs, took 0.06 GiB
init engine (profile, create kv cache, warmup model) took 340.60 seconds
Started server process [129894]
```

Runtime import/JIT validation has **not** been completed locally because this
local `.venv` does not have `torch` installed.

Remote runtime progress after syncing Option A:

- First remote serve attempt reached CUDA graph profiling and failed in
  `triton_deepseek_v4_sparse_decode()` because the output tensor had fewer
  heads than the query tensor (`assert out.shape == q.shape`). Patched the
  Triton sparse wrappers to assert only token/head-dim compatibility and slice
  `q`/`out` to the shared head count.
- Second remote serve attempt reached Triton compilation and failed because
  `_DSV4_HEAD_DIM` was a Python module global read from inside `@triton.jit`.
  Patched the kernel to use JIT-visible `tl.constexpr(...)` constants for
  DeepSeek V4 dimensions.
- Third remote serve attempt reached CUDA graph memory profiling and then
  failed with `cudaErrorIllegalAddress`, reported while the sparse indexer
  fallback synchronized on `decode_lens.sum().item()`. The traceback was in
  `model.layers.2`, which suggests either the indexer cache path failed or an
  earlier asynchronous attention kernel reported late.
- Found a concrete bug in the Triton sparse decode wrapper: DeepSeek V4's
  FlashMLA path passes decode `q` as `[tokens, 1, heads, dim]`, but the Triton
  wrapper treated dimension 1 as the head axis. Patched
  `triton_deepseek_v4_sparse_decode()` to squeeze that singleton dimension and
  compute all heads instead of only head 0.
- Synced `vllm/v1/attention/ops/triton_mla_sparse.py` to
  `/home/ross/DS/Lvllm/vllm/v1/attention/ops/triton_mla_sparse.py`.
  Local and remote SHA256 both equal
  `9dc22306b053c05fef7d5c697fb31b1b7e0cf2e184d7c4f89802a618f18b2334`.

2026-04-29 4096-token runtime request failure:

- User ran:

```bash
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=20 \
  OMP_NUM_THREADS=20 vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --config deepseek_v4.yaml --cpu-offload-gb 142 \
  --gpu-memory-utilization 0.95 --max-model-len 4096 \
  --max-num-batched-tokens 4096
```

- Server initialized successfully, but the first chat request crashed with:

```text
repeat_interleave output_size argument (1) must be the same as the sum of the
elements in the repeats tensor (18)
```

- Scheduler dump showed one cached request with `num_computed_tokens=[18]` and
  one scheduled decode token. The sparse-indexer fallback was mixing scheduled
  decode-token count with total cached context-token count.
- Local patches applied:
  - `vllm/model_executor/layers/sparse_attn_indexer.py`
    - `_topk_per_row_decode_fallback()` now derives total cached length and
      decode-token count from `seq_lens_cpu` / `decode_lens_cpu`.
    - `_gather_indexer_k_quant_cache_torch()` accepts `seq_lens_cpu` and keeps
      `repeat_interleave(output_size=...)` aligned with the repeat sum.
    - Follow-up capture fix: the fallback no longer constructs
      `torch.tensor(seq_lens_cpu, device=...)` during CUDA graph capture; it
      reuses the existing GPU `seq_lens` tensor and only uses CPU metadata for
      Python scalar totals.
  - `vllm/v1/attention/backends/mla/indexer.py`
    - Multi-token decode flattening now builds `decode_lens_for_expand` from
      the CPU lengths before `repeat_interleave`, avoiding CPU/GPU metadata
      mismatch.
  - `vllm/v1/attention/backends/mla/flashmla_sparse.py`
    - C128A metadata now uses `cm.seq_lens_cpu` instead of synchronizing
      `cm.seq_lens.cpu()` in the sparse metadata builder.
- Local verification:

```bash
.venv/bin/python -m py_compile \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/v1/attention/backends/mla/indexer.py \
  vllm/v1/attention/backends/mla/flashmla_sparse.py

git diff --check -- \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/v1/attention/backends/mla/indexer.py \
  vllm/v1/attention/backends/mla/flashmla_sparse.py
```

- First remote rerun after the repeat-size patch moved past the original
  `Repeat.cu` assertion and exposed the CUDA graph capture allocation issue:

```text
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
```

- The local follow-up patch addresses that capture issue, but remote sync and
  rerun were not completed after context compaction because the SSH credential
  was no longer available to the agent.
- Remote `.venv/bin/python` does not exist, so remote `py_compile` was not run.
  Local `py_compile` and `git diff --check` passed for the synced kernel file.

2026-04-29 rerun evidence:

- The new traceback is more precise: startup still fails during CUDA graph
  memory profiling, but the observed root stack is inside
  `DeepseekV4Indexer.forward()` before `self.mla_attn(...)` is called.
- The first confirmed crash site is the DeepGEMM-missing CUDA fallback in
  `vllm/model_executor/layers/sparse_attn_indexer.py`, specifically after
  `ops.cp_gather_indexer_k_quant_cache(...)` launches and a later `.item()`
  synchronizes.
- This does **not** yet prove the new Triton sparse decode kernel is bad; the
  sparse indexer cache gather crashes first.
- Patched `_topk_per_row_decode_fallback()` to bypass
  `cp_gather_indexer_k_quant_cache` and gather the same packed indexer cache
  bytes with PyTorch tensor indexing. The helper preserves the packed page
  layout used by `indexer_k_quant_and_cache`: all FP8 K bytes first, then scale
  bytes.
- Local verification:

```bash
.venv/bin/python -m py_compile \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/v1/attention/ops/triton_mla_sparse.py

git diff --check -- \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/v1/attention/ops/triton_mla_sparse.py
```

Both commands passed.

2026-04-29 later rerun evidence:

- The previous illegal CUDA memory access is gone.
- Startup now progresses through model loading, compile, initial profiling,
  and into CUDA graph memory profiling.
- New root failure is a normal Python metadata mismatch:

```text
AttributeError: 'FlashMLASparseMetadata' object has no attribute
'c128a_global_decode_topk_indices'
```

- The failing path is `DeepseekV4MLAAttention._forward_decode()` for
  `compress_ratio == 128`. C128A layers intentionally do not have a
  `DeepseekV4Indexer`, so they rely on the FlashMLA sparse metadata builder to
  precompute deterministic compressed top-k indices.
- Patched `vllm/v1/attention/backends/mla/flashmla_sparse.py` to add and
  populate:
  - `c128a_global_decode_topk_indices`
  - `c128a_decode_topk_lens`
  - `c128a_prefill_topk_indices`
- Local verification:

```bash
.venv/bin/python -m py_compile \
  vllm/v1/attention/backends/mla/flashmla_sparse.py \
  vllm/model_executor/layers/deepseek_v4_attention.py \
  vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py

git diff --check -- \
  vllm/v1/attention/backends/mla/flashmla_sparse.py \
  vllm/model_executor/layers/deepseek_v4_attention.py \
  continue.md
```

Both commands passed.

Next concrete action: sync `flashmla_sparse.py` to the remote and rerun the
serve smoke test:

```bash
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=20 \
  OMP_NUM_THREADS=20 vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --config deepseek_v4.yaml --cpu-offload-gb 138
```

If it fails again, inspect the new root traceback first. The next most likely
issue is in the C128A metadata shape/range assumptions or the new Triton sparse
fp8 decode path.

2026-04-29 remote smoke after C128A patch:

- Synced `flashmla_sparse.py` and verified remote hash:
  `710b4006c66640a9b19f47c108297d8434d5e892c74e3c9dae25805cbfb3c7e5`.
- The rerun passed the previous
  `c128a_global_decode_topk_indices` AttributeError.
- New failure occurs during CUDA graph memory profiling/capture:

```text
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
```

- Root call site:

```text
vllm/model_executor/layers/sparse_attn_indexer.py
  _topk_per_row_decode_fallback
  total_seq_lens = int(seq_lens.sum().item())
```

- This is the DeepGEMM-unavailable PyTorch fallback path for the sparse indexer.
  CUDA graph capture forbids the host synchronization from `.item()`.
- Patched the indexer metadata builder to store CPU decode lengths at metadata
  build time and patched `_topk_per_row_decode_fallback()` to use those Python
  constants during capture. Also updated the PyTorch cache gather helper to use
  `torch.repeat_interleave(..., output_size=...)` and tensor masking, avoiding
  `.item()`/`valid.all().item()` during capture.
- Local verification:

```bash
.venv/bin/python -m py_compile \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/v1/attention/backends/mla/indexer.py

git diff --check -- \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/v1/attention/backends/mla/indexer.py
```

Both commands passed.

---

## Files Touched / Relevant

- `cmake/external_projects/flashmla.cmake`
- `vllm/v1/attention/backends/mla/flashmla_sparse.py` (SM120 now excluded)
- `vllm/v1/attention/ops/flashmla.py`
- `vllm/platforms/cuda.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/backends/mla/triton_mla_sparse.py`
- `vllm/v1/attention/ops/triton_mla_sparse.py`
- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/model_executor/layers/deepseek_v4_attention.py`
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
