# DeepSeek-V4-Flash Serving Handoff Log

Last updated: 2026-04-28

## Goal

Bring up `deepseek-ai/DeepSeek-V4-Flash` with `vllm serve` on the remote DS box, using aggressive CPU offload and LK-MoE where possible, without crashing on consumer-GPU / incomplete custom-op paths.

## Environment

| Item | Value |
|---|---|
| Local repo | `/Users/ross/Documents/project/Lvllm` |
| Remote repo | `/home/ross/DS/Lvllm` |
| Remote host | `ross@192.168.1.16` |
| Remote env | `conda activate DS` |
| Model | `deepseek-ai/DeepSeek-V4-Flash` |
| Primary launch pattern | `LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=20 OMP_NUM_THREADS=20 vllm serve deepseek-ai/DeepSeek-V4-Flash --config deepseek_v4.yaml --cpu-offload-gb <N>` |
| Observed working load config | `--cpu-offload-gb 138` with `gpu_memory_utilization: 0.65` in `deepseek_v4.yaml` gets through weight load and compile |

## Current State

- The latest remote run gets past:
  - model weight load
  - MXFP4 fallback path selection
  - `torch.compile`
- The latest hard failure is no longer an OOM. It moved forward to another missing custom op in the Marlin MXFP4 fallback path:

```text
AttributeError: '_OpNamespace' '_moe_C' object has no attribute 'moe_wna16_marlin_gemm'
```

- That op is required for the only CUDA MXFP4 backend that appears usable on this consumer GPU path.
- Local code now guards Marlin backend selection and the wrapper call so this fails early with a clear "rebuild `_moe_C`" message instead of after a long model load.
- The next step is to verify/rebuild the remote CUDA `_moe_C` extension, then rerun the serve command.

## Findings

### 1. Rope setup for DeepSeek-V4 was incorrect for this kernel path

The DeepSeek V4 attention path expected:
- `rotary_dim == rope_head_dim == 64`
- `cos_sin_cache.dtype == torch.float32`

Before patching, the model was building RoPE with:
- effective rotary dim 512 instead of 64
- `bfloat16` cache instead of `float32`

That caused assertion failures in `fused_inv_rope_fp8_quant`.

### 2. `deep_gemm` is not usable on this remote GPU for the MHC path

The remote machine is on a consumer GPU, not Hopper/Blackwell. The `tf32_hc_prenorm_gemm` fast path is therefore unavailable or outdated there. A PyTorch fallback is required for profile/dummy run to proceed.

### 3. LK-MoE initialization for MXFP4 cannot always materialize dense weights safely

With `DeepSeek-V4-Flash` and high CPU offload, the MXFP4 LK initialization path may try to materialize dense BF16 expert weights and blow through host-memory expectations. We saw repeated warnings like:

```text
Skipping lk_moe init for MXFP4 layer ... dense weights need 12.00 GiB and init peak is estimated at 36.00 GiB ...
```

This is acceptable as long as the code preserves the packed MXFP4 path and falls back cleanly to Marlin.

### 4. The remote `_moe_C` build is incomplete

Confirmed missing on the remote DS environment:

- `topk_softplus_sqrt`
- `moe_align_block_size`
- `batched_moe_align_block_size`
- `moe_sum`
- `moe_permute`
- `moe_unpermute`
- `grouped_topk`
- `moe_wna16_marlin_gemm`

This means the remote setup cannot assume the full custom MoE op surface is present. Fallbacks are necessary on any execution path that touches these symbols.

The latest failure shows that `moe_wna16_marlin_gemm` is not optional for the selected MXFP4 Marlin backend. Unlike `moe_align_block_size` or `moe_sum`, this GEMM cannot be reasonably replaced by a small Python fallback after weights have been converted into Marlin layout. The remote native extension needs to be rebuilt or replaced with one that contains CUDA MoE Marlin support.

### 5. The failure progression is now improving

The debugging sequence moved from:
- import / kernel preconditions
- to rope assertions
- to missing `deep_gemm`
- to GPU OOM during Marlin/LK-MoE setup
- to clean model load + compile
- to missing `_moe_C.moe_align_block_size`

That is a good sign: the runtime is getting meaningfully farther each iteration.

## Amendments Applied

### A. DeepSeek-V4 rope fixes

File: `vllm/model_executor/models/deepseek_v4.py`

Applied:
- set `partial_rotary_factor = rope_head_dim / head_dim`
- force RoPE cache dtype to `torch.float32`

Effect:
- fixes the `rotary_dim` mismatch
- fixes the `cos_sin_cache.dtype == torch.float32` assertion

### B. `deep_gemm` fallback for MHC pre-norm

File: `vllm/utils/deep_gemm.py`

Applied:
- added a pure PyTorch fallback in `tf32_hc_prenorm_gemm(...)`

Effect:
- avoids hard failure when the remote GPU cannot use the `deep_gemm` implementation

### C. MXFP4 LK-MoE memory-aware fallback

Files:
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py`

Applied:
- initialize `self.lk_moe` / `self.lk_moe_config` safely
- add memory-aware checks before trying to materialize dense BF16 expert weights for LK-MoE
- skip LK init when host-memory peak would be unsafe
- preserve packed MXFP4 weights so Marlin fallback can still run
- guard runner LK usage with `getattr(layer, "lk_moe", None) is not None`

Effect:
- stops the code from destroying the fallback path when LK init is skipped
- allows the model to proceed on the default MXFP4 / Marlin route

### D. `sqrtsoftplus` router fallback

File: `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`

Applied:
- guard usage of `_moe_C.topk_softplus_sqrt`
- when the op is missing, fall back to the generic PyTorch scoring path instead of throwing

Effect:
- removes the hard stop on the remote build where `topk_softplus_sqrt` is absent

### E. `moe_align_block_size` and `batched_moe_align_block_size` fallbacks

File: `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`

Applied:
- added pure-PyTorch fallbacks for:
  - `moe_align_block_size`
  - `batched_moe_align_block_size`
- wired them behind runtime `hasattr(torch.ops._moe_C, ...)` checks

Effect:
- removes the latest hard crash on missing `_moe_C.moe_align_block_size`
- keeps Marlin/MoE routing working even without the custom op

### F. `moe_sum` fallback

File: `vllm/_custom_ops.py`

Applied:
- when `_moe_C.moe_sum` is missing, fall back to:

```python
output.copy_(input.sum(dim=1))
```

Effect:
- preempts the next likely failure on the same incomplete `_moe_C` build

### G. Remote dependency adjustment

Remote environment:
- `tilelang` was installed in the DS env so the MHC import path can load

### H. MXFP4 Marlin op availability guard

Files:
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
- `vllm/_custom_ops.py`

Applied:
- treat Marlin/Batched Marlin as unsupported when `_moe_C.moe_wna16_marlin_gemm` is missing
- raise a direct rebuild-oriented `RuntimeError` if the wrapper is still called without the op

Effect:
- prevents the current remote setup from spending several minutes loading and compiling before failing with a raw `AttributeError`
- clarifies that this blocker requires a matching CUDA `_moe_C` build, not another Python fallback

## Verification Completed

### Local checks

- `py_compile` passed for the new fallback files after patching

### Remote checks

- synced patched files from local repo to `/home/ross/DS/Lvllm/...`
- confirmed the remote `_moe_C` namespace is missing several expected symbols
- validated the pure-PyTorch `moe_align_block_size` fallback against the docstring example
- validated the `moe_sum` fallback against `input.sum(dim=1)`

Key verified results:
- align fallback produced the expected sorted-token ordering
- `moe_sum` fallback matched the reference reduction

## Latest Remote Failure Before the Last Patch

From the `--cpu-offload-gb 140` run:

```text
AttributeError: '_OpNamespace' '_moe_C' object has no attribute 'moe_wna16_marlin_gemm'
```

That exact missing-op path is now guarded, but serving still requires a rebuilt `_moe_C` extension containing the op.

## 2026-04-28 Remote Rebuild Result

The "No MXFP4 MoE backend supports the deployment configuration" failure was
the expected fail-fast path from the new backend guard. The real underlying
problem was that importing the remote `_moe_C` extension failed with:

```text
ImportError: /home/ross/DS/Lvllm/vllm/_moe_C.abi3.so: undefined symbol: _Z18topk_softplus_sqrtRN2at6TensorES1_S1_S1_bdSt8optionalIS0_ES3_S3_
```

Root cause: `csrc/moe/moe_ops.h` declared the optional `topk_softplus_sqrt`
arguments by value, while `csrc/moe/topk_softplus_sqrt_kernels.cu` defined
them as `const c10::optional<torch::Tensor>&`. The declaration was updated to
match the definition.

Remote rebuild command used in the DS conda environment:

```bash
/home/ross/.local/bin/uv pip install -e . --torch-backend=auto --no-build-isolation
```

Rebuild result:

```text
Built vllm @ file:///home/ross/DS/Lvllm
Installed vllm==0.1.dev15691+g5c6838540.d20260428.cu132
```

Post-rebuild verification on `ross@192.168.1.16`:

```text
moe_C file /home/ross/DS/Lvllm/vllm/_moe_C.abi3.so
has moe_wna16_marlin_gemm True
has topk_softplus_sqrt True
```

The rebuilt extension timestamp was:

```text
2026-04-28 09:24:44.000000000 +0800 96994000 vllm/_moe_C.abi3.so
```

Bounded serve smoke test:

```bash
timeout 180s bash -lc "LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=20 OMP_NUM_THREADS=20 vllm serve deepseek-ai/DeepSeek-V4-Flash --config deepseek_v4.yaml --cpu-offload-gb 140"
```

Result:

```text
Using 'MARLIN' Mxfp4 MoE backend.
Loading weights took 98.15 seconds
Skipping lk_moe init for MXFP4 layer model.layers.0.ffn.experts: dense weights need 12.00 GiB and init peak is estimated at 36.00 GiB, but only 40.43 GiB host RAM is currently available. Falling back to the default MXFP4 MoE path.
```

The smoke test was intentionally terminated by `timeout` before the API server
could keep running. No leftover `vllm serve`, `APIServer`, or `EngineCore`
processes remained afterward.

## Next Action

Rerun the remote serve command:

```bash
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=20 OMP_NUM_THREADS=20 \
vllm serve deepseek-ai/DeepSeek-V4-Flash --config deepseek_v4.yaml --cpu-offload-gb 140
```

If it still fails, capture the next traceback. The current `_moe_C` import and
Marlin op registration issue is fixed on the remote server. The remaining LK MoE
warning is a host-RAM headroom issue during LK init, not the missing Marlin op
failure.

If backend selection fails again, re-check `_moe_C` import directly in the DS
environment:

```bash
source /home/ross/miniconda3/etc/profile.d/conda.sh
conda activate DS
/home/ross/.local/bin/uv run --active --no-project python - <<'PY'
import importlib
import torch
mod = importlib.import_module("vllm._moe_C")
print(mod.__file__)
print("has moe_wna16_marlin_gemm:", hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"))
print("has topk_softplus_sqrt:", hasattr(torch.ops._moe_C, "topk_softplus_sqrt"))
PY
```

## Do Not

- Do not revert the MXFP4 LK fallback guards in `layer.py`; they are what preserve the Marlin fallback path.
- Do not assume the remote `_moe_C` extension is feature-complete.
- Do not treat the current local git worktree as clean; there are unrelated modified files present.
- Do not overwrite the remote repo blindly without checking for new local edits first.

## Files Touched In This Debug Track

- `vllm/_custom_ops.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`
- `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`
- `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py`
- `vllm/model_executor/models/deepseek_v4.py`
- `vllm/utils/deep_gemm.py`
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
- `csrc/moe/moe_ops.h`

## Local Worktree Notes

Current `git status --short` also shows other modified files outside the narrow fallback patches, including:

- `vllm/model_executor/layers/deepseek_v4_attention.py`
- `vllm/model_executor/layers/quantization/mxfp4.py`
- `vllm/v1/kv_cache_interface.py`

Treat those as existing in-flight changes unless separately reviewed. Do not revert them as part of this handoff.

## 2026-04-28 Local Follow-Up: DeepSeek V4 KV Cache Page Sizes

After the `_moe_C.moe_wna16_marlin_gemm` registration issue was fixed, the
server progressed through Marlin backend selection, model load, torch compile,
and warmup, then failed during CUDA graph KV cache profiling:

```text
NotImplementedError: The page size of the layer is not divisible by the maximum page size. Cannot unify by adjusting block_size.
```

The failure is in `vllm/v1/core/kv_cache_utils.py` while unifying hybrid KV cache
page sizes. DeepSeek V4 creates several cache specs with different aligned page
sizes, including main MLA/SWA cache pages, C128 compressed pages, indexer cache
pages, and compressor state pages. These are not clean integer multiples of the
largest page, so the existing compact hybrid allocator cannot normalize them by
only increasing `block_size`.

Local-only patch applied:

- `vllm/v1/core/kv_cache_utils.py`
  - Fall back to heterogeneous per-layer KV tensor allocation when page sizes
    are not divisible.
  - Account for heterogeneous page sizes in block-count and max-memory
    estimates.
- `vllm/v1/worker/gpu_model_runner.py`
  - Reshape MLA-family caches using compressed storage block sizes.
  - Use `torch.as_strided` for padded cache pages so alignment padding does not
    have to appear in the logical backend cache shape.

Verification run locally:

```bash
.venv/bin/python -m py_compile \
  vllm/v1/core/kv_cache_utils.py \
  vllm/v1/worker/gpu_model_runner.py

git diff --check -- \
  vllm/v1/core/kv_cache_utils.py \
  vllm/v1/worker/gpu_model_runner.py
```

Both commands passed. This has not been synced to `192.168.1.16`.

## 2026-04-28 Local Follow-Up: DeepSeek V4 MLA Dimension Fields

After the heterogeneous KV cache fallback was synced, startup progressed past
the KV page-size unification error and failed while constructing FlashMLA sparse
metadata:

```text
AttributeError: 'DeepseekV4Config' object has no attribute 'kv_lora_rank'. Did you mean: 'o_lora_rank'?
```

`get_mla_dims()` was assuming older MLA config field names. DeepSeek V4 stores
the equivalent latent KV width as `head_dim`, and the DeepSeek V4 attention
wrapper already passes `kv_lora_rank=self.head_dim`.

Local patch applied:

- `vllm/model_executor/layers/attention/mla_attention.py`
  - `get_mla_dims()` now falls back to `head_dim` when `kv_lora_rank` is absent.
  - `qk_nope_head_dim` is derived as `kv_lora_rank - qk_rope_head_dim` when the
    explicit field is absent.
  - `v_head_dim` falls back to `kv_lora_rank` when absent.

Verification run locally:

```bash
.venv/bin/python -m py_compile \
  vllm/model_executor/layers/attention/mla_attention.py
```

This command passed.

## 2026-04-28 Local Follow-Up: Experimental SM120 FlashMLA Port

After the CPU-offload profiling patch was synced, startup reached CUDA graph
memory profiling and then failed while building DeepSeek V4 sparse MLA metadata:

```text
RuntimeError: vllm._flashmla_C is not available, likely was not compiled due to insufficient nvcc version or a supported arch was not in the list of target arches to compile for.
```

The target server reports an RTX PRO 2000 Blackwell GPU with compute capability
`(12, 0)`. The existing FlashMLA build gate only listed Hopper `9.0a` and
SM10x Blackwell targets (`10.0a` / `10.0f`), while other vLLM kernels already
list SM12x targets.

Local experimental port applied:

- `cmake/external_projects/flashmla.cmake`
  - Added SM12x FlashMLA target arches: `12.0f` for CUDA 13+ and
    `12.0a` / `12.1a` for CUDA 12.8+.
- `vllm/v1/attention/ops/flashmla.py`
  - Split sparse availability from dense-extension availability so sparse
    FlashMLA only requires `_flashmla_C`.
  - Allowed SM120-family devices through `is_flashmla_sparse_supported()`.
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`
  - Allowed compute-capability major `12` for the FlashMLA sparse backend.
- `vllm/platforms/cuda.py`
  - Treat MLA on major `12` similarly to major `10` for backend priority order.

Local verification run:

```bash
.venv/bin/python -m py_compile \
  vllm/v1/attention/ops/flashmla.py \
  vllm/v1/attention/backends/mla/flashmla_sparse.py \
  vllm/platforms/cuda.py
```

This command passed. The next required verification is a remote rebuild with
`CUDA_HOME=/usr/local/cuda-12.9` and `/usr/local/cuda-12.9/bin` on `PATH`; the
FlashMLA source may still need deeper SM120 kernel changes if the upstream
SM100 instantiations do not compile or run for SM120.

## 2026-04-28 Local Follow-Up: FlashMLA SM120 Runtime Dispatch

After rebuilding FlashMLA on the Blackwell host, both extension modules imported
cleanly, but a real sparse decode smoke test failed with:

```text
RuntimeError: Unsupported architecture for sparse decode fwd
```

Root cause: upstream FlashMLA's `csrc/api/common.h` reports `Arch::is_sm100f()`
only when CUDA device major is `10`. RTX PRO 2000 Blackwell reports compute
capability `12.0`, so it passed the vLLM Python/CMake gates but was rejected by
FlashMLA's C++ sparse decode dispatcher before launching the SM100 sparse
decode kernels.

Local patch applied:

- `cmake/external_projects/flashmla.cmake`
  - Patches the fetched FlashMLA `csrc/api/common.h` during CMake configure so
    `is_sm100f()` accepts major `12` as well as major `10`.

The next verification is rebuilding `_flashmla_C` on the server and rerunning
the sparse decode smoke test.

## 2026-04-28 Handoff: Remote Rebuild In Progress

The SM120 runtime-dispatch patch was synced to the server and applied to the
already-fetched FlashMLA source:

```text
.deps/flashmla-src/csrc/api/common.h
bool is_sm100f() const {
    return major == 10 || major == 12;
}
```

The editable rebuild was restarted on the server with:

```bash
cd /home/ross/DS/Lvllm
source /home/ross/miniconda3/etc/profile.d/conda.sh
conda activate DS
export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
export PATH=/usr/local/cuda-12.9/bin:$PATH
export MAX_JOBS=24
/home/ross/.local/bin/uv pip install -e . --torch-backend=auto --no-build-isolation -v
```

As of 2026-04-28 20:25 HKT, the remote build was still active:

```text
uv pip install PID 63436
cmake --build PID 64120
ninja PID 64121
build tree /tmp/tmpaycba69p.build-temp
```

See `continue.md` for the exact sparse FlashMLA smoke test to run after this
build exits.

## 2026-04-28 Local Follow-Up: Profiling InputBatch Reinit With Offload

After the MLA dimension compatibility patch was synced, startup progressed past
metadata-builder construction and failed during minimal KV-cache CUDA graph
profiling:

```text
AssertionError: Cannot re-initialize the input batch when CPU weight offloading is enabled.
```

The temporary profiling KV cache now has the real DeepSeek V4 heterogeneous
block layout, which differs from the pre-load placeholder input batch. The
regular runtime path still should not rebuild the input batch after CPU offload,
but the profiling path needs matching block tables before graph warmup/capture.

Local patch applied:

- `vllm/v1/worker/gpu_model_runner.py`
  - `may_reinitialize_input_batch()` now accepts `is_profiling`.
  - The CPU-offload reinit guard is relaxed only for profiling initialization.
  - `initialize_kv_cache()` forwards its existing `is_profiling` flag.

Verification run locally:

```bash
.venv/bin/python -m py_compile \
  vllm/v1/worker/gpu_model_runner.py
```

This command passed.

## 2026-04-28 Local Follow-Up: FlashMLA SM120 Kernel Dispatch — SM100 → SM90

After the second `_flashmla_C` rebuild (with the `is_sm100f()` SM120 patch), the
sparse decode smoke test ran the kernel but failed with:

```text
CUDA error (/home/ross/DS/Lvllm/.deps/flashmla-src/csrc/sm100/decode/head64/
instantiations/../kernel.cuh:956): CUDA error: invalid argument
```

Root cause: `is_sm100f()` returning true for SM120 caused the C++ dispatcher to
select the SM100 sparse decode kernel path. The SM100 kernels use CUDA
persistent-cluster launch configurations that SM120 (GB202, consumer Blackwell)
does not support — `cudaLaunchKernelEx` returns `cudaErrorInvalidValue` when the
cluster geometry is invalid for the device.

SM120 can run SM90 (Hopper) kernels via PTX forward compatibility. SM90 sparse
decode kernels use smaller, per-device cluster configurations that SM120 supports.
The Python metadata already uses the SM90 formula for SM120 (`is_device_capability_family(100)`
returns False for major==12, so both `max_num_sm_parts` and `prefill_padding` take
the SM90 branch).

Local patch applied:

- `cmake/external_projects/flashmla.cmake`
  - Removed the `is_sm100f()` SM120 patch.
  - Added a new patch: `is_sm90f()` now accepts major `9` or `12`, routing SM120
    to the SM90 sparse decode kernel path.

Before the next rebuild the already-fetched `common.h` on the remote must be
reset to its original state, otherwise the new `string(REPLACE ...)` targeting
`"return major == 9;"` will find no match:

```bash
cd /home/ross/DS/Lvllm/.deps/flashmla-src
git checkout csrc/api/common.h
grep -n "major ==" csrc/api/common.h   # verify is_sm90f says "return major == 9;"
```
