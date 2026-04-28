# Continue - DeepSeek V4 Flash / Blackwell FlashMLA

## Last Action

Synced and applied a FlashMLA SM120 runtime-dispatch patch, then restarted the remote editable rebuild. As of 2026-04-28 20:25 HKT, the server build is still running: `uv` PID `63436`, CMake PID `64120`, Ninja PID `64121`, build tree `/tmp/tmpaycba69p.build-temp`.

## Next Action

Poll the remote build without interrupting it. When `uv pip install` exits, run the sparse FlashMLA smoke test below on the server:

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

## Why

The previous extension rebuild imported successfully, but an actual Blackwell sparse decode call failed with `RuntimeError: Unsupported architecture for sparse decode fwd`. The local CMake patch now rewrites upstream FlashMLA `csrc/api/common.h` so `Arch::is_sm100f()` accepts device major `12` as well as `10`.

## Current Changes

- `cmake/external_projects/flashmla.cmake`: adds SM12x FlashMLA build targets and patches upstream FlashMLA `common.h` at configure time.
- `vllm/v1/attention/ops/flashmla.py`: separates sparse FlashMLA availability from dense-extension availability and allows sparse SM120.
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`: allows compute capability major `12`.
- `vllm/platforms/cuda.py`: treats major `12` like major `10` for MLA backend priority.
- `vllm/model_executor/layers/attention/mla_attention.py`: derives missing DeepSeek V4 MLA dimension fields.
- `vllm/v1/worker/gpu_model_runner.py`: permits profiling-only input-batch reinit while CPU offload is enabled.
- `logs/dsv4_debug_log.md`: running notes for all failures and patches.

## Open Threads

- The remote rebuild is running with `MAX_JOBS=8`. This was conservative because previous template-heavy CUDA builds were memory-sensitive. Future rebuilds can try a higher value if RAM headroom is healthy.
- Dense FlashMLA still reports unsupported on Blackwell. That is separate from DeepSeek V4 sparse SWA acceleration; do not treat the dense gate as the current blocker.
- If the sparse smoke still fails after this rebuild, inspect the new exception before changing Python gates again. The likely next layer would be SM100 kernel assumptions running on SM120, not import or backend selection.

## Do Not

- Do not kill the current remote build unless the user asks.
- Do not sync or rerun a full install over the current build until it exits.
- Do not put the SSH password in notes or commits.
