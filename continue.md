# Continue - DeepSeek V4 Flash / Blackwell FlashMLA

## Last Action

Smoke-tested the second `_flashmla_C` build (with the `is_sm100f()` SM120 patch).
The extension imported and the C++ dispatcher accepted SM120, but the actual SM100
sparse decode kernel failed at launch:

```text
CUDA error (.../csrc/sm100/decode/head64/instantiations/../kernel.cuh:956):
CUDA error: invalid argument
```

Root cause: SM100 kernels use CUDA persistent-cluster launch geometry that SM120
(consumer Blackwell GB202) does not support.

New local fix: `cmake/external_projects/flashmla.cmake` now patches `is_sm90f()`
to accept major `9 || 12` instead of patching `is_sm100f()`. SM90 kernels work on
SM120 via PTX forward compatibility and use cluster configs SM120 supports. The
Python metadata (max_num_sm_parts formula, prefill_padding) already uses the SM90
path for SM120.

## Next Action

### Step 1 — Reset common.h on the remote

```bash
cd /home/ross/DS/Lvllm/.deps/flashmla-src
git checkout csrc/api/common.h
grep -n "major ==" csrc/api/common.h
```

Expected output — both functions must be in their original form:

```
XX:        return major == 9;
YY:        return major == 10;
```

If they are still patched (e.g. `major == 10 || major == 12`), force-reset:

```bash
git stash   # or: git checkout HEAD -- csrc/api/common.h
```

### Step 2 — Sync updated cmake file

```bash
rsync -av /Users/ross/Documents/project/Lvllm/cmake/external_projects/flashmla.cmake \
  ross@192.168.1.16:/home/ross/DS/Lvllm/cmake/external_projects/flashmla.cmake
```

Or pull via git if the commit is pushed to origin.

### Step 3 — Rebuild _flashmla_C

```bash
cd /home/ross/DS/Lvllm
source /home/ross/miniconda3/etc/profile.d/conda.sh
conda activate DS
export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
export PATH=/usr/local/cuda-12.9/bin:$PATH
export MAX_JOBS=24
/home/ross/.local/bin/uv pip install -e . --torch-backend=auto --no-build-isolation
```

### Step 4 — Verify the patch landed

```bash
grep -n "major ==" /home/ross/DS/Lvllm/.deps/flashmla-src/csrc/api/common.h
```

Expected after rebuild:

```
XX:        return major == 9 || major == 12;   ← is_sm90f patched
YY:        return major == 10;                  ← is_sm100f unchanged
```

### Step 5 — Re-run sparse smoke test

```bash
cd /home/ross/DS/Lvllm
source /home/ross/miniconda3/etc/profile.d/conda.sh
conda activate DS
/home/ross/.local/bin/uv run --active --no-project python - <<'PY'
import sys, torch
sys.path.insert(0, "/home/ross/DS/Lvllm/.deps/flashmla-src/tests")
import vllm.third_party.flashmla.flash_mla_interface as flash_mla
sys.modules["flash_mla"] = flash_mla
from lib import RawTestParamForDecode
import lib

p = RawTestParamForDecode(
    b=4, h_q=64, s_q=1, h_kv=1, s_kv=512, is_varlen=True,
    topk=64, block_size=64, d_qk=576, check_correctness=False,
    num_runs=0, seed=0,
).to_test_param()
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda:0")
torch.cuda.set_device("cuda:0")
t = lib.generate_testcase_for_decode(p)
meta, splits = flash_mla.get_mla_metadata()
out, lse = lib.run_flash_mla_decode(p, t, meta, splits)
torch.cuda.synchronize()
print("out", out.shape, out.dtype, "lse", lse.shape, lse.dtype)
print("sched", meta.tile_scheduler_metadata.shape, meta.num_splits.shape)
PY
```

Success looks like:
```
out torch.Size([4, 64, 512]) torch.bfloat16 lse torch.Size([4, 64, 1]) torch.bfloat16
sched torch.Size([N, 8]) torch.Size([5])
```

### Step 6 — If smoke test passes, run full serve

```bash
LVLLM_MOE_NUMA_ENABLED=1 LK_THREAD_BINDING=CPU_CORE LK_THREADS=20 OMP_NUM_THREADS=20 \
  vllm serve deepseek-ai/DeepSeek-V4-Flash --config deepseek_v4.yaml --cpu-offload-gb 140
```

## Why

SM90 sparse decode kernels are compiled for sm_90a and are JIT-compatible on
SM120. The cluster-launch parameters SM90 uses (per-SM parts computed via
`sm_count // (h_q/64)`) are within SM120's supported cluster dimensions. The SM100
path uses persistent block-cluster configurations that SM120's driver rejects.

## Current Changes

- `cmake/external_projects/flashmla.cmake`: patches `is_sm90f()` to accept major 12
  (SM90 kernel path for SM120), removed the earlier `is_sm100f()` SM120 patch.
- `vllm/v1/attention/ops/flashmla.py`: allows sparse FlashMLA on SM120.
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`: allows compute capability major 12.
- `vllm/platforms/cuda.py`: treats major 12 like major 10 for MLA backend priority.
- `vllm/model_executor/layers/attention/mla_attention.py`: derives missing DeepSeek V4 MLA dims.
- `vllm/v1/worker/gpu_model_runner.py`: permits profiling-only input-batch reinit with CPU offload.
- `vllm/v1/core/kv_cache_utils.py`: heterogeneous KV page-size fallback.
- `logs/dsv4_debug_log.md`: running notes for all failures and patches.

## Open Threads

- If the smoke test still fails after routing SM120 to SM90, the next step is to
  inspect what SM90 kernel dispatch returns (e.g. `is_sm90f()` might use a
  different condition string in the original source). Check via
  `grep -n "is_sm90\|major ==" .deps/flashmla-src/csrc/api/common.h`.
- Dense FlashMLA is still SM90-only; this is separate from sparse SWA.
- After the serve is stable, the `moe_sum` / `moe_align` Python fallbacks should
  be profiled to confirm they are not on the hot path.

## Do Not

- Do not restore the `is_sm100f()` SM120 patch; it causes kernel launch failure.
- Do not assume `common.h` is unpatched on the remote; always `git checkout` it
  before triggering a cmake-patching rebuild.
- Do not put the SSH password in notes or commits.
