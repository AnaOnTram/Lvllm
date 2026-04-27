# Adding mxfp4 support for DeepSeek V4 in vLLM NUMA

**Session ID:** ses_233a3f559ffe7yKFQ6Ib5sBi0z
**Created:** 4/27/2026, 8:34:37 AM
**Updated:** 4/27/2026, 9:04:52 AM

---

## User

Think about how to add DeepSeek V4 Pro support to this vllm NUMA plugin. The final intention is to run the deepseek v4 flash on a system with 192GB RAM and 16GB VRAM (RTX PRO 2000). The know issue is some of the parameter is some of the model parameters are quantized in mxfp4 which current plugin does not support.

---

## Assistant (Build · DeepSeek V4 Pro · 262.1s)

_Thinking:_

The user wants to add DeepSeek V4 Pro support to a vLLM NUMA plugin. Let me first understand what this project is about and find the relevant NUMA plugin code.

Key context:
- This is a vLLM project (likely vllm-project/vllm based on the AGENTS.md)
- They want to add DeepSeek V4 Pro support to a NUMA plugin
- Target system: 192GB RAM, 16GB VRAM (RTX PRO 2000)
- Issue: Some model parameters are quantized in mxfp4 which current plugin doesn't support

Let me explore the codebase to understand:
1. What the NUMA plugin is
2. Where quantization formats are handled
3. What mxfp4 is and where it would need to be supported

Let me start by exploring the project structure.

**Tool: task**

**Input:**
```json
{
  "description": "Explore vLLM NUMA plugin",
  "prompt": "Explore the vLLM codebase at /Users/ross/Documents/project/Lvllm to understand:\n\n1. Find the NUMA plugin - search for files related to \"numa\" (filename and content). Read the main NUMA plugin files.\n2. Find where quantization formats (like fp4, mxfp4, fp8, nf4, etc.) are defined and handled. Look for quantization-related constants, enums, or config classes.\n3. Find where model loading and memory planning happens - look for files related to \"worker\", \"model_runner\", \"gpu_executor\", \"memory\".\n4. Look for any existing DeepSeek model support (search for \"deepseek\" in filenames and content).\n5. Look for any \"mxfp4\" or \"mx\" format references.\n\nReturn:\n- The full paths to all relevant files found\n- A summary of how the NUMA plugin works (what directory it's in, what it does)\n- How quantization formats are currently handled (what formats are supported)\n- Any existing DeepSeek model support\n- The architecture of model loading/pipeline parallelism",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_233a39f9effezL3Re2pxfZdNwC (for resuming to continue this task if needed)

<task_result>
Now I have a comprehensive picture of the codebase. Here is my complete report:

---

## Final Report: vLLM Codebase Exploration

---

### 1. NUMA Plugin

There is **no standalone NUMA plugin directory or file**. NUMA support is implemented inline across a few files:

**Relevant files:**
- `/Users/ross/Documents/project/Lvllm/vllm/v1/executor/multiproc_executor.py` (lines 690-694, 1049-1076)
- `/Users/ross/Documents/project/Lvllm/vllm/envs.py` (lines 1946-1956)
- `/Users/ross/Documents/project/Lvllm/vllm/platforms/cpu.py` (lines 41-65, 81, 135-136, 345-351, 407-409)

**How it works:**

The NUMA support is controlled by the environment variable `LVLLM_ENABLE_NUMA_INTERLEAVE` (checked via `is_numa_interleave_enabled()` in `envs.py`). When enabled, the `MultiprocExecutor` (the multi-process distributed executor at `/Users/ross/Documents/project/Lvllm/vllm/v1/executor/multiproc_executor.py`) intercepts worker process creation:

1. When starting a worker process (`WorkerProc.make_proc_handle`, line 690), if `is_numa_interleave_enabled()` is true, it creates a temporary shell script wrapper (`_create_numactl_executable()`, line 1049) that prepends `numactl --interleave=all` to the Python executable.
2. This wrapper script is then set as the multiprocessing spawn executable (`multiprocessing.spawn.set_executable()`), so all worker processes are launched under `numactl --interleave=all`, which ensures memory allocations are interleaved across all NUMA nodes.

Additionally, in `platforms/cpu.py`, the `CpuPlatform` class has NUMA topology discovery (`discover_numa_topology()`) and simulation capabilities for multi-NUMA testing. The `LogicalCPUInfo` dataclass tracks `numa_node` per logical CPU. There is also a `_SIM_MULTI_NUMA` environment variable for testing.

---

### 2. Quantization Formats

**Central definition file:**
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/__init__.py`

**Currently supported quantization methods** (from the `QuantizationMethods` Literal at line 12):

| Quantization Method | Config Class File |
|---|---|
| `awq` | `awq.py` |
| `fp8` | `fp8.py` |
| `fbgemm_fp8` (deprecated) | `fbgemm_fp8.py` |
| `fp_quant` (deprecated) | `fp_quant.py` |
| `modelopt` | `modelopt.py` (`ModelOptFp8Config`) |
| `modelopt_fp4` | `modelopt.py` (`ModelOptNvFp4Config`) |
| `modelopt_mxfp8` | `modelopt.py` (`ModelOptMxFp8Config`) |
| `modelopt_mixed` | `modelopt.py` (`ModelOptMixedPrecisionConfig`) |
| `gguf` | `gguf.py` |
| `gptq_marlin` | `gptq_marlin.py` |
| `awq_marlin` | `awq_marlin.py` |
| `gptq` | `gptq.py` |
| `compressed-tensors` | `compressed_tensors/compressed_tensors.py` |
| `bitsandbytes` | `bitsandbytes.py` |
| `experts_int8` (deprecated) | `experts_int8.py` |
| `quark` | `quark/quark.py` |
| `moe_wna16` | `moe_wna16.py` |
| `torchao` | `torchao.py` |
| `inc` / `auto-round` | `inc.py` |
| `mxfp4` | `mxfp4.py` |
| `mxfp8` | `mxfp8.py` |
| `cpu_awq` | `cpu_wna16.py` |
| `online` | `online/base.py` |
| `fp8_per_tensor` / `fp8_per_block` | (shorthand for online quant) |

**Quantization infrastructure folder:**
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/`
  - `base_config.py` — `QuantizationConfig` (ABC) and `QuantizeMethodBase`
  - `__init__.py` — `QUANTIZATION_METHODS`, `register_quantization_config()`, `get_quantization_config()`
  - `utils/` — helper utilities (`quant_utils.py`, `mxfp4_utils.py`, `mxfp6_utils.py`, `mxfp8_utils.py`, `fp8_utils.py`)
  - `compressed_tensors/` — CompressedTensors integration
  - `quark/` — AMD Quark quantization schemes
  - `online/` — Online (dynamic) quantization

---

### 3. MXFP4 and MX Format References

**Extensive MX format support exists across these files:**

MXFP4 quantization (weight/activation):
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/mxfp4.py` — `Mxfp4Config` (QuantizationConfig), `Mxfp4MoEMethod`
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py` — Swizzle/scale layout utilities
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` — `Mxfp4MoeBackend` enum (NONE, FLASHINFER_TRTLLM_MXFP4_MXFP8, FLASHINFER_TRTLLM_MXFP4_BF16, FLASHINFER_CUTLASS_MXFP4_MXFP8, FLASHINFER_CUTLASS_MXFP4_BF16, BATCHED_MARLIN, MARLIN, AITER, TRITON)
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py`
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py`
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py`

MXFP8 quantization:
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/mxfp8.py` — `Mxfp8Config` (extends `Fp8Config`), `Mxfp8OnlineLinearMethod`, `Mxfp8OnlineMoEMethod`
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/fused_moe/oracle/mxfp8.py`
- `/Users/ross/Documents/project/Lvllm/csrc/moe/mxfp8_moe/` — CUDA kernels

MXFP6 utilities:
- `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/utils/mxfp6_utils.py`

Tests:
- `/Users/ross/Documents/project/Lvllm/tests/quantization/reference_mxfp4.py`
- `/Users/ross/Documents/project/Lvllm/tests/models/quantization/test_mxfp4.py`
- `/Users/ross/Documents/project/Lvllm/tests/models/quantization/test_mxfp8.py`
- `/Users/ross/Documents/project/Lvllm/tests/kernels/quantization/test_mxfp4_triton_ep.py`
- `/Users/ross/Documents/project/Lvllm/tests/kernels/quantization/test_mxfp4_qutlass.py`
- `/Users/ross/Documents/project/Lvllm/tests/kernels/moe/test_cutlass_mxfp8_grouped_mm.py`

Evaluation configs referencing mxfp4:
- 7 YAML files in `/Users/ross/Documents/project/Lvllm/tests/evals/gpt_oss/configs/`

---

### 4. DeepSeek Model Support

**DeepSeek model files in `/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/`:**

| File | Purpose |
|---|---|
| `deepseek_v2.py` (1694 lines) | Main implementation: DeepSeek V2/V3 architecture (MLA attention, MoE, PP support, LoRA, Eagle) |
| `deepseek_mtp.py` | Multi-Token Prediction (MTP) spec decoding |
| `deepseek_eagle.py` | Eagle speculative decoding for DeepSeek V2 |
| `deepseek_eagle3.py` | Eagle3 speculative decoding for DeepSeek V3 |
| `deepseek_vl2.py` | Vision-Language model (DeepSeek VL2) |
| `deepseek_ocr.py` | OCR model |
| `deepseek_ocr2.py` | Second version OCR model |

**Other DeepSeek-related files:**
- Tool parsers: `deepseekv3_tool_parser.py`, `deepseekv31_tool_parser.py`, `deepseekv32_tool_parser.py`
- Tokenizers: `deepseek_v32.py`, `deepseek_v32_encoding.py`
- Reasoning parsers: `deepseek_v3_reasoning_parser.py`, `deepseek_r1_reasoning_parser.py`
- Rope embedding: `rotary_embedding/deepseek_scaling_rope.py`
- Renderers: `deepseek_v32.py`
- Configs/processors: `transformers_utils/configs/deepseek_vl2.py`, `transformers_utils/processors/deepseek_vl2.py`, `transformers_utils/processors/deepseek_ocr.py`
- Chat templates: `template_deepseek_vl2.jinja`, `template_deepseek_ocr.jinja`
- Examples: `ray_serve_deepseek.py`, `serve_deepseek_v2.sh`
- CI/Build: `deepseek_v2_lite_ep_eplb.sh`, `deepseek_v2_lite_prefetch_offload.sh`

**Architecture support in `deepseek_v2.py`:**
- `DeepseekV2ForCausalLM` inherits from: `nn.Module`, `SupportsPP` (pipeline parallelism), `DeepseekV2MixtureOfExperts`, `SupportsLoRA`, `SupportsEagle`, `SupportsEagle3`
- Supports both DeepSeek V2 and V3 configs
- Supports MLA (Multi-head Latent Attention), standard MHA, and sparse attention (V3.2 indexer)
- Models: `DeepseekForCausalLM`, `DeepseekV3ForCausalLM`, `GlmMoeDsaForCausalLM`

---

### 5. Architecture of Model Loading / Pipeline Parallelism

**Model loading pipeline (top to bottom):**

1. **`Executor`** (`/Users/ross/Documents/project/Lvllm/vllm/v1/executor/abstract.py`) — Abstract base class. Factory method `get_class()` selects:
   - `MultiprocExecutor` (multi-process, default non-Ray)
   - `RayDistributedExecutor` / `RayExecutorV2` (Ray-based)
   - `UniProcExecutor` / `ExecutorWithExternalLauncher`

2. **`Worker`** (`/Users/ross/Documents/project/Lvllm/vllm/v1/worker/gpu_worker.py`) — Each worker process:
   - Calls `init_device()` → initializes CUDA, distributed env (NCCL), sets random seed, takes memory snapshot
   - Creates a **`ModelRunner`** (V1 or V2): `GPUModelRunner` from `gpu_model_runner.py` (V1) or `v1/worker/gpu/model_runner.py` (V2)
   - Calls `load_model()` → delegates to `model_runner.load_model()`
   - Performs memory profiling in `determine_available_memory()` to compute available KV cache memory

3. **`ModelLoader`** (`/Users/ross/Documents/project/Lvllm/vllm/model_executor/model_loader/__init__.py`) — Dispatches to format-specific loaders:
   - `DefaultModelLoader` (HF/safetensors)
   - `BitsAndBytesModelLoader`, `GGUFModelLoader`, `TensorizerLoader`, `ShardedStateLoader`, `RunaiModelStreamerLoader`, `DummyModelLoader`
   - `BaseModelLoader.load_model()`: initializes model structure via `initialize_model()`, then calls `load_weights()`, then `process_weights_after_loading()` (where quantization happens), then optionally `finalize_layerwise_processing()` for online quant.

4. **Model initialization** uses `make_layers()`/`SupportsPP`:
   - Models declare `start_layer`/`end_layer` via `make_layers()` (from `models/utils.py`)
   - Each PP rank only creates its assigned layers; others use `PPMissingLayer` placeholder
   - `SupportsPP` interface (from `interfaces.py`) handles `IntermediateTensors` passing between PP stages

**Key files in the pipeline:**

| File | Role |
|---|---|
| `/Users/ross/Documents/project/Lvllm/vllm/v1/executor/abstract.py` | Executor ABC, factory for executor types |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/executor/multiproc_executor.py` | Multi-process executor, NUMA support |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/executor/uniproc_executor.py` | Single-process executor |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/executor/ray_executor.py` | Ray distributed executor (V1) |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/executor/ray_executor_v2.py` | Ray distributed executor (V2) |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/worker/worker_base.py` | `WorkerBase` and `WorkerWrapperBase` |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/worker/gpu_worker.py` | GPU `Worker` class (init_device, load_model, memory profiling) |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/worker/gpu_model_runner.py` | GPU ModelRunner V1 (391+ lines, `GPUModelRunner`, `AsyncGPUModelRunnerOutput`) |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/worker/gpu/model_runner.py` | GPU ModelRunner V2 (103+ lines, `GPUModelRunner` with `LoRAModelRunnerMixin`) |
| `/Users/ross/Documents/project/Lvllm/vllm/model_executor/model_loader/__init__.py` | Model loader registry, `get_model()` entry point |
| `/Users/ross/Documents/project/Lvllm/vllm/model_executor/model_loader/base_loader.py` | `BaseModelLoader` (initialize → load_weights → process_weights → finalize) |
| `/Users/ross/Documents/project/Lvllm/vllm/model_executor/model_loader/default_loader.py` | Default HF/safetensors loader |
| `/Users/ross/Documents/project/Lvllm/vllm/model_executor/model_loader/utils.py` | `initialize_model()`, `get_model_architecture()` |
| `/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/interfaces.py` | Model interfaces: `SupportsPP`, `SupportsLoRA`, `SupportsEagle`, `MixtureOfExperts`, `VllmModel` |
| `/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/utils.py` | `make_layers()`, `PPMissingLayer`, `is_pp_missing_parameter()` |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/worker/gpu/block_table.py` | `BlockTables` for KV cache block management |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/worker/block_table.py` | `BlockTable`, `MultiGroupBlockTable` |
| `/Users/ross/Documents/project/Lvllm/vllm/v1/kv_cache_interface.py` | `KVCacheSpec`, `KVCacheConfig`, `KVCacheQuantMode` |
</task_result>
```

---

## Assistant (Build · DeepSeek V4 Pro · 259.2s)

_Thinking:_

Excellent! This is very informative. Let me now understand the situation better:

1. The NUMA plugin is actually inline in the multiproc_executor.py - it adds `numactl --interleave=all` wrapper for worker processes
2. **MXFP4 is already supported** in vLLM! There's an `mxfp4.py` quantization config, `mxfp4_utils.py`, MoE backends like `FLASHINFER_TRTLLM_MXFP4_MXFP8`, `FLASHINFER_CUTLASS_MXFP4_BF16`, `BATCHED_MARLIN`, `MARLIN`, `AITER`, `TRITON`, etc.
3. DeepSeek V2/V3 models are already supported via `deepseek_v2.py`

So the user says "mxfp4 which current plugin does not support" - but the exploration shows that mxfp4 IS supported in the vLLM codebase. The issue might be:
- The quantization method string used for loading
- The specific flavor of mxfp4 used by DeepSeek V4 Pro
- The NUMA plugin specifically not handling mxfp4 weights

Wait, let me re-read the user's message: "The final intention is to run the deepseek v4 flash on a system with 192GB RAM and 16GB VRAM (RTX PRO 2000)."

DeepSeek V4 Pro / DeepSeek V4 Flash - this is a new model. DeepSeek might have a new architecture or new quantization scheme.

Let me look more carefully at:
1. The quantization config for mxfp4 to understand how it's registered
2. What the `modelopt_fp4` config does (since it's the fp4 quantization from modelopt)
3. Whether there's a distinction between `mxfp4` and `modelopt_fp4`
4. How models declare their quantization method

The key issue might be that DeepSeek V4 Pro uses a new quantization format name in its model config that vLLM doesn't recognize yet.

Let me dig deeper into the specific files.

**Tool: task**

**Input:**
```json
{
  "description": "Deep dive mxfp4 and quant config",
  "subagent_type": "explore",
  "prompt": "Very thorough exploration of the vLLM codebase at /Users/ross/Documents/project/Lvllm. I need to understand the full picture of MXFP4 and modelopt_fp4 support, and how DeepSeek V4 Pro could be added.\n\n1. Read the full contents of:\n   - /Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/mxfp4.py\n   - /Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/modelopt.py (this contains ModelOptFp8Config, ModelOptNvFp4Config, ModelOptMxFp8Config, ModelOptMixedPrecisionConfig)\n   - /Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/__init__.py (the QuantizationMethods literal and how quantization is registered)\n\n2. Read the DeepSeek model to see how it declares quantization:\n   - /Users/ross/Documents/project/Lvllm/vllm/model_executor/models/deepseek_v2.py (focus on quant_config, quantization_method, any fp4 or mxfp4 references)\n\n3. Look at how models declare their quantization method - search for \"quantization_config\" or \"quant_method\" in model config files. How does a model tell vLLM what quantization to use?\n\n4. Check the modelopt.py for the ModelOptNvFp4Config - does it use mxfp4 under the hood?\n\n5. Search for \"nvfp4\" or \"nv_fp4\" across the codebase (NVIDIA's FP4 format)\n\n6. Search for \"modelopt\" usage in deepseek_v2.py\n\n7. Look at the quantization config registry - how are new quantization methods registered? What's the QUANTIZATION_METHODS dict?\n\n8. Read the mxfp4_utils.py:\n   - /Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py\n\nReturn ALL the content you find and a thorough summary."
}
```

**Output:**
```
task_id: ses_2339f874effeXDtg2tQYTnclGD (for resuming to continue this task if needed)

<task_result>
Now I have all the information needed. Here is the comprehensive summary:

---

## COMPLETE ANALYSIS: MXFP4 and ModelOpt FP4 Support in vLLM

### 1. `mxfp4.py` -- The Standalone MXFP4 Config

**File:** `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/mxfp4.py` (413 lines)

- **`Mxfp4Config(QuantizationConfig)`**: The quantization config class for MXFP4.
  - `get_name()` returns `"mxfp4"` -- this is the vLLM internal quantization method name.
  - `get_min_capability()` returns `80` (Ampere+).
  - `get_supported_act_dtypes()` returns `[torch.bfloat16]`.
  - `get_config_filenames()` returns `[]` -- no auto-detection; it must be specified manually by the user (e.g., `quantization="mxfp4"`).
  - **`get_quant_method()`**: For `LinearBase` layers, it falls back to `UnquantizedLinearMethod()` (MXFP4 linear layers are NOT implemented). For `FusedMoE` layers, returns `Mxfp4MoEMethod`. For `Attention`, it logs a skip.
  - `is_mxfp4_quant()` always returns `True`.

- **`Mxfp4MoEMethod(FusedMoEMethodBase)`**: The quant method for MXFP4 MoE.
  - Stores weights as `torch.uint8` (packed 4-bit), scales as `torch.uint8`.
  - Block size = 32.
  - Creates `w13_weight`, `w13_weight_scale`, `w2_weight`, `w2_weight_scale` parameters.
  - Selects a backend via `select_mxfp4_moe_backend()` from `oracle/mxfp4.py`.
  - Backends include: FLASHINFER_TRTLLM (BF16 and MXFP8), FLASHINFER_CUTLASS (BF16 and MXFP8), MARLIN, BATCHED_MARLIN, AITER (ROCm), TRITON, XPU.
  - Uses modular kernel architecture: `make_mxfp4_moe_kernel()` creates a `FusedMoEKernel` that handles execution.

- **`Mxfp4Config` does NOT have an `override_quantization_method()`** -- it inherits the base that returns `None`. This means **MXFP4 cannot be auto-detected from checkpoint configs**; the user must explicitly pass `--quantization mxfp4`.

---

### 2. `modelopt.py` -- ModelOpt Quantization (FP8, NVFP4, MXFP8, Mixed)

**File:** `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/modelopt.py` (2187 lines)

This file contains ALL the ModelOpt quantization configs. Here's the architecture:

#### Base Classes
- **`QuantizationConfig`** (abstract base in `base_config.py`):
  - Abstract methods: `get_name()`, `get_supported_act_dtypes()`, `get_min_capability()`, `get_config_filenames()`, `from_config()`, `get_quant_method()`.
  - Non-abstract: `override_quantization_method()` (default returns `None`), `is_mxfp4_quant()` (default returns `False`), `packed_modules_mapping`.

- **`ModelOptQuantConfigBase(QuantizationConfig)`**: Shared base for all modelopt configs.
  - Attributes: `LinearMethodCls`, `FusedMoEMethodCls`, `KVCacheMethodCls` (set by subclasses).
  - Handles `exclude_modules` with wildcard matching.
  - `get_quant_method()` delegates to the appropriate class method based on layer type.
  - `from_config()` parses both legacy (`hf_quant_config.json` with `quantization` dict) and new (`config.json` compressed-tensors style) formats.
  - `override_quantization_method()` is **NOT** defined here -- subclasses define their own.
  - `_extract_modelopt_quant_algo()` static method extracts `quant_algo` from configs.

#### `ModelOptFp8Config` -- FP8 Quantization
- `get_name()` returns `"modelopt"`.
- `get_min_capability()` = 89.
- `override_quantization_method()`: If `quant_algo == "FP8"`, returns `"modelopt"`.
- Supports three modes via `quant_method`: `"FP8"`, `"FP8_PER_CHANNEL_PER_TOKEN"`, `"FP8_PB_WO"`.
- Each mode has its own LinearMethod (`ModelOptFp8LinearMethod`, `ModelOptFp8PcPtLinearMethod`, `ModelOptFp8PbWoLinearMethod`).
- MoE uses `ModelOptFp8MoEMethod` with FlashInfer TRTLLM FP8 backend.

#### `ModelOptNvFp4Config` -- NVIDIA FP4 Quantization
- `get_name()` returns `"modelopt_fp4"`.
- `get_min_capability()` = 75.
- `override_quantization_method()`: If quant_algo contains `"NVFP4"` or `"FP4"`, returns `"modelopt_fp4"`.
- Requires `is_checkpoint_nvfp4_serialized` to be True.
- **Linear method: `ModelOptNvFp4LinearMethod`** -- Handles dense linear layers with NVFP4.
  - Backend auto-selection via `select_nvfp4_linear_backend()`: CUTLASS, FlashInfer-CUTLASS, FlashInfer-TRTLLM, FlashInfer-CUDNN, FBGEMM, MARLIN, EMULATION.
  - Weight format: `uint8` packed (2 FP4 values per byte), scales in `float8_e4m3fn`, plus global scales (`input_scale`, `weight_scale_2`).
  - Group size = 16 (default, configurable).
- **MoE method: `ModelOptNvFp4FusedMoE`** -- Handles MoE layers with NVFP4.
  - Backends via `select_nvfp4_moe_backend()`: FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, FLASHINFER_CUTEDSL, VLLM_CUTLASS, MARLIN.
  - Creates `w13_weight`, `w2_weight` (packed uint8), `w13_weight_scale`, `w2_weight_scale` (float8_e4m3fn block scales), plus per-tensor `weight_scale_2` and `input_scale`.
  - Supports `use_global_sf` for expert parallelism.

#### `ModelOptMxFp8Config` -- Micro-scaling FP8 Quantization
- `get_name()` returns `"modelopt_mxfp8"`.
- `get_min_capability()` = 80.
- `override_quantization_method()`: If `"MXFP8"` in quant_algo, returns `"modelopt_mxfp8"`.
- Uses FP8 E4M3 values with E8M0 scales, block size 32.
- MoE uses FlashInfer TRTLLM MXFP8 block-scale MoE.

#### `ModelOptMixedPrecisionConfig` -- Mixed FP8 + NVFP4
- `get_name()` returns `"modelopt_mixed"`.
- `override_quantization_method()`: If `quant_algo == "MIXED_PRECISION"`, returns `"modelopt_mixed"`.
- Contains both `fp8_config` and `nvfp4_config` as child configs.
- `get_quant_method()` resolves per-layer `quant_algo` from a `quantized_layers` dict in the config.
- Routes layers to either FP8 or NVFP4 methods based on the per-layer algo.

---

### 3. `__init__.py` -- Quantization Registration and Registry

**File:** `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/__init__.py` (191 lines)

- **`QuantizationMethods`** (Literal type): Lists ALL valid quantization method names:
  ```python
  "awq", "fp8", "fbgemm_fp8", "fp_quant", "modelopt", "modelopt_fp4",
  "modelopt_mxfp8", "modelopt_mixed", "gguf", "gptq_marlin", "awq_marlin",
  "gptq", "compressed-tensors", "bitsandbytes", "experts_int8", "quark",
  "moe_wna16", "torchao", "inc", "mxfp4", "mxfp8", "cpu_awq", "online",
  "fp8_per_tensor", "fp8_per_block"
  ```

- **`QUANTIZATION_METHODS`**: A list of all method names from the Literal.

- **`get_quantization_config(quantization: str)`**: The main registry. Maps string names to config classes:
  ```python
  "modelopt" -> ModelOptFp8Config
  "modelopt_fp4" -> ModelOptNvFp4Config
  "modelopt_mxfp8" -> ModelOptMxFp8Config
  "modelopt_mixed" -> ModelOptMixedPrecisionConfig
  "mxfp4" -> Mxfp4Config
  "mxfp8" -> Mxfp8Config
  ```

- **`register_quantization_config(quantization: str)`**: Decorator to register custom/plugin quantization methods into `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`.

---

### 4. How Quantization Method is Auto-Detected

**File:** `/Users/ross/Documents/project/Lvllm/vllm/config/model.py` (lines 929-999)

The `_verify_quantization()` method:
1. Reads `quantization_config` from the model's HF config file (either `config.json`'s `quantization_config` field or `hf_quant_config.json`).
2. Gets `quant_method` from the config (e.g., `"modelopt"`).
3. Iterates over all known quantization methods, calling `override_quantization_method(hf_quant_cfg, user_quant)` on each config class.
4. The **overrides list** defines priority order:
   ```python
   overrides = ["gptq_marlin", "awq_marlin", "inc", "moe_wna16",
                "modelopt", "modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed",
                "mxfp4", "cpu_awq", "gguf"]
   ```
5. Each override method inspects the checkpoint's `quant_algo` and returns the vLLM method name to use (or `None`).

**For modelopt checkpoints**, the flow is:
- `hf_quant_config.json` has `{"quantization": {"quant_algo": "NVFP4", ...}}`
- `ModelOptNvFp4Config.override_quantization_method()` extracts `quant_algo`, sees `"NVFP4"`, returns `"modelopt_fp4"`.
- vLLM sets `quantization="modelopt_fp4"` and instantiates `ModelOptNvFp4Config`.

---

### 5. Does `ModelOptNvFp4Config` Use MXFP4 Under the Hood?

**NO.** NVFP4 and MXFP4 are completely separate FP4 formats with different:
- **Block sizes**: NVFP4 uses 16, MXFP4 uses 32.
- **Scale dtypes**: NVFP4 uses `float8_e4m3fn` for block scales, MXFP4 uses `uint8` (E8M0).
- **Additional scales**: NVFP4 has per-tensor global scales (`weight_scale_2`, `input_scale`); MXFP4 does not.
- **Backend/oracle modules**: 
  - NVFP4 MoE: `oracle/nvfp4.py` (selects backends: FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, etc.)
  - MXFP4 MoE: `oracle/mxfp4.py` (selects backends: FLASHINFER_TRTLLM_MXFP4, MARLIN, AITER, XPU, TRITON, etc.)
- **Linear kernels**: NVFP4 uses `cutlass_scaled_fp4_mm`/FlashInfer; MXFP4 linear layers are NOT supported (falls back to unquantized).
- **Config classes**: `ModelOptNvFp4Config` (name `"modelopt_fp4"`) vs `Mxfp4Config` (name `"mxfp4"`).

They are orthogonal implementations serving different NVIDIA quantization pipelines.

---

### 6. DeepSeek V2 Model and Quantization

**File:** `/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/deepseek_v2.py` (1694 lines)

DeepSeek V2 uses generic quantization through `QuantizationConfig`:

- **`DeepseekAttention`**, **`DeepseekV2Attention`**, **`DeepseekV2MLAAttention`**, **`DeepseekV2MLP`**: All accept `quant_config: QuantizationConfig | None` and pass it to their linear layers (e.g., `QKVParallelLinear`, `MergedColumnParallelLinear`, `RowParallelLinear`, `ColumnParallelLinear`, `ReplicatedLinear`).

- **`DeepseekV2MoE`**: Accepts `quant_config: QuantizationConfig | None`, passes it to `SharedFusedMoE` (which is actually `FusedMoE` under the hood).

- **`Indexer.__init__`**: Has specific logic for `modelopt_fp4`:
  ```python
  self.is_fp4_ckpt = (
      self.quant_config is not None
      and self.quant_config.get_name() == "modelopt_fp4"
  )
  ```
  When `is_fp4_ckpt` is True, it fuses `wk` + `weights_proj` into a single `MergedColumnParallelLinear` with `quant_config=None` (since `weights_proj` is not quantized).

- **`DeepseekV2ForCausalLM.__init__`**: Same `is_fp4_ckpt` check, extends `stacked_params_mapping` with the indexer fused mapping for weight loading.

- **`DeepseekV2ForCausalLM.load_weights()`**: The `is_fp4_ckpt` flag determines if `indexer_fused_mapping` entries are added to `stacked_params_mapping` for remapping `wk_weights_proj` weights.

- **No references to `mxfp4` in deepseek_v2.py**. The model only has special handling for `modelopt_fp4` (NVFP4).

---

### 7. How `SharedFusedMoE` / `FusedMoE` Processes Quantization

**File:** `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/fused_moe/layer.py` (2776 lines)

The `FusedMoE.__init__()` flow:
1. Stores `quant_config` (can be `None`).
2. Creates `moe_config: FusedMoEConfig`.
3. Calls `_get_quant_method()`:
   ```python
   quant_method = None
   if self.quant_config is not None:
       quant_method = self.quant_config.get_quant_method(self, prefix)
   if quant_method is None:
       quant_method = UnquantizedFusedMoEMethod(self.moe_config)
   self.quant_method: FusedMoEMethodBase = quant_method
   ```
4. Calls `self.quant_method.maybe_roundup_sizes()` to adjust hidden/intermediate sizes.
5. Calls `self.quant_method.create_weights()` to create all weight parameters.
6. A `DefaultMoERunner` wraps the layer with the quant method for execution.

So the quant method controls weight creation, rounding, and execution.

---

### 8. MXFP4 Utilities

**File:** `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py` (164 lines)

Provides:
- `_swizzle_mxfp4(quant_tensor, scale, num_warps=8)`: Weight swizzling for MXFP4 MoE using `triton_kernels` (OAI kernel format). Transposes tensors, calls `convert_layout()`.
- `_dequant_mxfp4(x, scale, float_dtype)`: Dequantizes MXFP4 using `quark.torch.kernel.mx`.
- `_quant_dequant_mxfp4(x, scale_calculation_mode)`: Quantize-dequantize MXFP4 using `quark`.
- Both dequant functions are registered as custom ops (`torch.ops.vllm.dequant_mxfp4`, `torch.ops.vllm.quant_dequant_mxfp4`) for torch.compile compatibility.
- `CK_MXFP4_MOE_DIM_ALIGNMENT = 256`: Required alignment for CK's MXFP4 MoE GEMM kernels (intermediate_size divisible by 256 after TP split).

---

### 9. Key Backend/Backend Selection Files

| Oracle Module | Purpose |
|---|---|
| `fused_moe/oracle/nvfp4.py` | NVFP4 MoE backend selection (FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, etc.) |
| `fused_moe/oracle/mxfp4.py` | MXFP4 MoE backend selection (FLASHINFER_TRTLLM_MXFP4, MARLIN, AITER, TRITON, etc.) |
| `fused_moe/oracle/fp8.py` | FP8 MoE backend selection |
| `fused_moe/oracle/mxfp8.py` | MXFP8 MoE backend selection |

| Utility Module | Purpose |
|---|---|
| `quantization/utils/nvfp4_utils.py` | NVFP4 linear GEMM (backends, quant, apply for dense layers) |
| `quantization/utils/mxfp4_utils.py` | MXFP4 dequant and swizzle |
| `quantization/utils/flashinfer_fp4_moe.py` | NVFP4 MoE flashinfer weight preparation |
| `quantization/utils/marlin_utils_fp4.py` | FP4 Marlin scale processing for both NVFP4 and MXFP4 |
| `quantization/utils/nvfp4_emulation_utils.py` | NVFP4 reference quant for emulation mode |

---

### 10. Summary: How to Add DeepSeek V4 Pro Support

If DeepSeek V4 Pro has a **new quant format**, you would need to:

1. **If using existing quantization** (e.g., modelopt NVFP4 or FP8): No changes needed -- the existing `DeepseekV2ForCausalLM` already passes `quant_config` to all layers, and the `Indexer` has special handling for `modelopt_fp4`. Just point `--quantization modelopt_fp4` or load a modelopt checkpoint that auto-detects.

2. **If adding a new quantization type**: 
   - Add a new `QuantizationConfig` subclass (like `ModelOptNvFp4Config`).
   - Register it in `__init__.py` with a `get_name()` value and add it to the `QuantizationMethods` Literal.
   - Implement `override_quantization_method()` for auto-detection from checkpoints.
   - If the new quant affects the DeepSeek indexer (`wk`/`weights_proj` fusion), add similar `is_fp4_ckpt` checks in `Indexer.__init__` and `DeepseekV2ForCausalLM`.

3. **If DeepSeek V4 Pro has architectural changes** (e.g., new attention, new MoE routing, new expert structure): Those would require model code changes regardless of quantization.

The key architectural decision point is: **NVFP4 (modelopt_fp4) and MXFP4 are different FP4 formats**. DeepSeek V2 currently only has special model-level handling for `modelopt_fp4` (NVFP4), not for `mxfp4`. If DeepSeek V4 Pro uses MXFP4 (or MXFP8), model-level changes may be needed for the indexer fusion logic.
</task_result>
```

---

## Assistant (Build · DeepSeek V4 Pro · 25.8s)

_Thinking:_

This is extremely valuable information. Let me now summarize the situation and think about what needs to be done:

## Current State

1. **mxfp4 IS already supported** in vLLM - both as a standalone `Mxfp4Config` and as part of modelopt via `ModelOptMxFp8Config`
2. **NVFP4 (modelopt_fp4) IS already supported** in vLLM
3. **DeepSeek V2/V3** models are already in the codebase
4. **DeepSeek V4 Pro** is NOT yet in the codebase

## What DeepSeek V4 Pro likely needs

Based on what I know:
- DeepSeek V4 Pro is a new model that may use MXFP4 quantization for some parameters
- The key issue is that the NUMA plugin (which is really the memory management / model loading pipeline) needs to handle the new model and its quantization format

But wait - the user said "the NUMA plugin does not support mxfp4." But the exploration shows mxfp4 IS supported. Let me think about what might really be missing:

1. The `mxfp4.py` config has `get_config_filenames()` returning `[]` - meaning it can't be auto-detected. Users must manually specify `--quantization mxfp4`.

2. But the bigger issue is probably: **DeepSeek V4 Pro is a brand new model** that vLLM doesn't yet support. It may use:
   - A different architecture than V2/V3
   - Different quantization that auto-detects as something vLLM doesn't recognize
   - MXFP4 for specific layers (attention, not just MoE)

3. The "NUMA plugin" reference might actually be about the model loading / memory management for the specific hardware constraints (192GB RAM + 16GB VRAM). The key challenge is running a large model on a small GPU, which requires:
   - CPU offloading (using system RAM for model weights)
   - Pipeline parallelism (splitting layers across devices)
   - Efficient quantization to fit in limited memory

Let me think about what I should present to the user as the analysis. The user asked me to "think about how to add DeepSeek V4 Pro support to this vllm NUMA plugin." Let me first research what DeepSeek V4 Pro actually is (its architecture, config format, etc.) to understand what's needed.

Actually wait, I should first ask the user clarifying questions about:
1. What they mean by "NUMA plugin" specifically - since NUMA support is inline in multiproc_executor.py
2. What exact DeepSeek V4 Pro model/checkpoint they have
3. What the specific error is when trying to run it

But before that, let me also look at the `from_config` / `override_quantization_method` flow more carefully to understand how a new model's quant config would be detected.

Actually, let me first research what DeepSeek V4 Pro / DeepSeek V4 Flash looks like in terms of its HuggingFace config, so I can give specific advice.

Now let me research what DeepSeek V4 Pro/Flash actually looks like to understand what specific changes are needed.

**Tool: tavily_tavily_search**

**Input:**
```json
{
  "query": "DeepSeek V4 Pro model architecture quantization format config.json",
  "search_depth": "advanced"
}
```

**Output:**
```
Detailed Results:

Title: config.json · deepseek-ai/DeepSeek-V4-Pro-Base at main
URL: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base/blob/main/config.json
Content: |  | "num\_experts\_per\_tok": 6, |
|  | "num\_hidden\_layers": 61, |
|  | "num\_hash\_layers": 3, |
|  | "num\_key\_value\_heads": 1, |
|  | "num\_nextn\_predict\_layers": 1, |
|  | "o\_groups": 16, |
|  | "o\_lora\_rank": 1024, |
|  | "q\_lora\_rank": 1536, |
|  | "qk\_rope\_head\_dim": 64, |
|  | "quantization\_config": { |
|  | "activation\_scheme": "dynamic", |
|  | "fmt": "e4m3", |
|  | "quant\_method": "fp8", |
|  | "scale\_fmt": "ue8m0", |
|  | "weight\_block\_size": [ |
|  | 128, |
|  | 128 |
|  | ] |
|  | }, |
|  | "rms\_norm\_eps": 1e-06, |
|  | "rope\_scaling": { |
|  | "beta\_fast": 32, |
|  | "beta\_slow": 1, |
|  | "factor": 16, |
|  | "original\_max\_position\_embeddings": 65536, |
|  | "type": "yarn" |
|  | }, |
|  | "rope\_theta": 10000, | [...] |  | }, |
|  | "rope\_theta": 10000, |
|  | "routed\_scaling\_factor": 2.5, |
|  | "scoring\_func": "sqrtsoftplus", |
|  | "sliding\_window": 128, |
|  | "swiglu\_limit": 10.0, |
|  | "tie\_word\_embeddings": false, |
|  | "topk\_method": "noaux\_tc", |
|  | "torch\_dtype": "bfloat16", |
|  | "transformers\_version": "4.57.1", |
|  | "use\_cache": true, |
|  | "vocab\_size": 129280, |
|  | "compress\_rope\_theta": 160000, |
|  | "compress\_ratios": [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0] |
|  | } |
|  |  | [...] |  |  |
 --- |
|  | { |
|  | "architectures": [ |
|  | "DeepseekV4ForCausalLM" |
|  | ], |
|  | "attention\_bias": false, |
|  | "attention\_dropout": 0.0, |
|  | "bos\_token\_id": 0, |
|  | "eos\_token\_id": 1, |
|  | "hc\_eps": 1e-06, |
|  | "hc\_mult": 4, |
|  | "hc\_sinkhorn\_iters": 20, |
|  | "head\_dim": 512, |
|  | "hidden\_act": "silu", |
|  | "hidden\_size": 7168, |
|  | "index\_head\_dim": 128, |
|  | "index\_n\_heads": 64, |
|  | "index\_topk": 1024, |
|  | "initializer\_range": 0.02, |
|  | "max\_position\_embeddings": 1048576, |
|  | "model\_type": "deepseek\_v4", |
|  | "moe\_intermediate\_size": 3072, |
|  | "n\_routed\_experts": 384, |
|  | "n\_shared\_experts": 1, |
|  | "norm\_topk\_prob": true, |
|  | "num\_attention\_heads": 128, |
|  | "num\_experts\_per\_tok": 6, |

Title: config.json · deepseek-ai/DeepSeek-V4-Pro at main
URL: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json
Content: |  | "num\_experts\_per\_tok": 6, |
|  | "num\_hidden\_layers": 61, |
|  | "num\_hash\_layers": 3, |
|  | "num\_key\_value\_heads": 1, |
|  | "num\_nextn\_predict\_layers": 1, |
|  | "o\_groups": 16, |
|  | "o\_lora\_rank": 1024, |
|  | "q\_lora\_rank": 1536, |
|  | "qk\_rope\_head\_dim": 64, |
|  | "quantization\_config": { |
|  | "activation\_scheme": "dynamic", |
|  | "fmt": "e4m3", |
|  | "quant\_method": "fp8", |
|  | "scale\_fmt": "ue8m0", |
|  | "weight\_block\_size": [ |
|  | 128, |
|  | 128 |
|  | ] |
|  | }, |
|  | "rms\_norm\_eps": 1e-06, |
|  | "rope\_scaling": { |
|  | "beta\_fast": 32, |
|  | "beta\_slow": 1, |
|  | "factor": 16, |
|  | "original\_max\_position\_embeddings": 65536, |
|  | "type": "yarn" |
|  | }, |
|  | "rope\_theta": 10000, | [...] |  |  |
 --- |
|  | { |
|  | "architectures": [ |
|  | "DeepseekV4ForCausalLM" |
|  | ], |
|  | "attention\_bias": false, |
|  | "attention\_dropout": 0.0, |
|  | "bos\_token\_id": 0, |
|  | "eos\_token\_id": 1, |
|  | "hc\_eps": 1e-06, |
|  | "hc\_mult": 4, |
|  | "hc\_sinkhorn\_iters": 20, |
|  | "head\_dim": 512, |
|  | "hidden\_act": "silu", |
|  | "hidden\_size": 7168, |
|  | "index\_head\_dim": 128, |
|  | "index\_n\_heads": 64, |
|  | "index\_topk": 1024, |
|  | "initializer\_range": 0.02, |
|  | "max\_position\_embeddings": 1048576, |
|  | "model\_type": "deepseek\_v4", |
|  | "moe\_intermediate\_size": 3072, |
|  | "n\_routed\_experts": 384, |
|  | "n\_shared\_experts": 1, |
|  | "norm\_topk\_prob": true, |
|  | "num\_attention\_heads": 128, |
|  | "num\_experts\_per\_tok": 6, | [...] |  | }, |
|  | "rope\_theta": 10000, |
|  | "routed\_scaling\_factor": 2.5, |
|  | "scoring\_func": "sqrtsoftplus", |
|  | "sliding\_window": 128, |
|  | "swiglu\_limit": 10.0, |
|  | "tie\_word\_embeddings": false, |
|  | "topk\_method": "noaux\_tc", |
|  | "torch\_dtype": "bfloat16", |
|  | "transformers\_version": "4.57.1", |
|  | "use\_cache": true, |
|  | "vocab\_size": 129280, |
|  | "compress\_rope\_theta": 160000, |
|  | "compress\_ratios": [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0] |
|  | } |
|  |  |

Title: DeepSeek V4 Pro API - Together AI
URL: https://www.together.ai/models/deepseek-v4-pro
Content: Try Nowread docs

## About model

DeepSeek V4 Pro is DeepSeek's 1.6T parameter (49B activated) MoE model supporting 1M token context. It introduces a hybrid attention architecture combining Compressed Sparse Attention and Heavily Compressed Attention, requiring only 27% of inference FLOPs and 10% of KV cache compared to V3.2 at million-token context. Pre-trained on 32T+ tokens with Muon optimizer and a two-stage post-training pipeline, V4 Pro delivers three configurable reasoning modes and strong performance across coding (93.5% LiveCodeBench), reasoning (90.1% GPQA Diamond), and agentic tasks (80.6% SWE-Bench Verified). MIT licensed.

Token Context

1M

27% FLOPs and 10% KV cache vs V3.2 at 1M context

LiveCodeBench

93.50%

Codeforces rating 3206 for competitive coding [...] Architecture Overview:   
  • 1.6T total parameter MoE architecture with 49B parameters activated per token   
  • Hybrid attention combining Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) for long-context efficiency   
  • At 1M token context: requires only 27% of single-token inference FLOPs and 10% of KV cache compared to DeepSeek-V3.2   
  • Manifold-Constrained Hyper-Connections (mHC) strengthening residual connections for stable signal propagation   
  • 1M token context window   
  • Three reasoning modes: Non-think (fast responses), Think High (logical analysis), Think Max (maximum reasoning effort)   
  • FP4 + FP8 mixed precision (MoE expert parameters in FP4, other parameters in FP8)   
    
  Training Methodology: [...] ‍

Related models

new

Chat

GLM-5.1

 Input/1M

  $1.40
 Output/1M

  $4.40
 Parameters

  754B

Function Calling

JSON Mode

Reasoning

Chat

New

Chat

MiniMax M2.7

 Input/1M

  $0.30
 Output/1M

  $1.20
 Parameters

  229B

Function Calling

JSON Mode

Prompt Caching

Reasoning

Code

Chat

new

Chat

Kimi K2.6

 Input/1M

  $1.20
 Output/1M

  $4.50
 Parameters

  1T

Function Calling

JSON Mode

Reasoning

Vision

Chat

Code

LLM

Model specificationsModel data

 Model provider

  

  DeepSeek
 Type

  Reasoning

  Chat

  Code

  LLM
 Main use cases

  Reasoning
 Features

  Function Calling

  JSON Mode
 Intelligence

  High
 Deployment

  Serverless

  Monthly Reserved
 Endpoint

  deepseek-ai/DeepSeek-V4-Pro
 Parameters

  1.6T
 Activated parameters

  49B
 Context length

Title: DeepSeek-V4: Towards Highly Efficient Million-Token Context ...
URL: https://arxiviq.substack.com/p/deepseek-v4-towards-highly-efficient
Content: At deployment, V4 relies heavily on FP4 (MXFP4) quantization. During training, FP32 master weights are quantized to FP4 and losslessly dequantized to FP8 for computation, allowing the model to adapt to low-precision inference natively. To manage the rigorous evaluation of its agentic capabilities, DeepSeek engineered DSec, a Rust-based, production-grade sandbox platform utilizing microVMs for secure, high-density, preemptible rollouts, ensuring that the evaluation of code-execution agents does not compromise the host cluster (Figure 5 illustrates the overlapping schemes required for this throughput).

### Empirical Footprint: Validating the Million-Token Horizon

Title: deepseek-ai/DeepSeek-V4-Pro
URL: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
Content: 1.   Hybrid Attention Architecture: We design a hybrid attention mechanism combining Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to dramatically improve long-context efficiency. In the 1M-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2.
2.   Manifold-Constrained Hyper-Connections (mHC): We incorporate mHC to strengthen conventional residual connections, enhancing stability of signal propagation across layers while preserving model expressivity.
3.   Muon Optimizer: We employ the Muon optimizer for faster convergence and greater training stability. [...] | MCPAtlas Public (Pass@1) | 73.8 | 67.2 | 69.2 | 66.6 | 71.8 | 73.6 |
| Toolathlon (Pass@1) | 47.2 | 54.6 | 48.8 | 50.0 | 40.7 | 51.8 | [...] We pre-train both models on more than 32T diverse and high-quality tokens, followed by a comprehensive post-training pipeline. The post-training features a two-stage paradigm: independent cultivation of domain-specific experts (through SFT and RL with GRPO), followed by unified model consolidation via on-policy distillation, integrating distinct proficiencies across diverse domains into a single model.
```

**Tool: tavily_tavily_search**

**Input:**
```json
{
  "query": "DeepSeek V4 Flash model huggingface config mxfp4 quantization",
  "search_depth": "advanced"
}
```

**Output:**
```
Detailed Results:

Title: amd/DeepSeek-OCR-MXFP4 · Hugging Face
URL: https://huggingface.co/amd/DeepSeek-OCR-MXFP4
Content: Hugging Face's logo

# amd / DeepSeek-OCR-MXFP4 like 0 Follow AMD 2.52k

# Model Overview

This model was built with DeepSeek-OCR model by applying AMD-Quark for MXFP4 quantization.

# Model Quantization

The model was quantized from amd/DeepSeek-OCR using AMD-Quark. The weights and activations are quantized to MXFP4.

Quantization scripts:

Before quantization, please install flash-attn in the following way:

`pip install flash-attn --no-build-isolation`

Note that deepseek\_vl\_v2 is not in the built-in model template list in Quark V0.11, it has to be registered before quantization. [...] `import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
from datasets import load_dataset
from quark.contrib.llm_eval import ppl_eval
# Register DeepSeek-OCR template
deepseek_ocr_template = LLMTemplate(
model_type="deepseek_vl_v2",
kv_layers_name=["k_proj", "v_proj"],
q_layer_name="q_proj",
exclude_layers_name=["lm_head", "model.sam_model", "model.vision_model", "model.projector"],
)
LLMTemplate.register_template(deepseek_ocr_template)
# Configuration
ckpt_path = "amd/DeepSeek-OCR"
output_dir = "amd/DeepSeek-OCR-MXFP4"
quant_scheme = "mxfp4"
exclude_layers = ["self_attn", "mlp.gate", "lm_head", "mlp.gate_proj", "mlp.up_proj", [...] "mlp.down_proj", "shared_experts.", "sam_model", "vision_model", "projector"]
# Load model
model = AutoModel.from_pretrained(ckpt_path, use_safetensors=True, trust_remote_code=True,
_attn_implementation='flash_attention_2', device_map="cuda:0", torch_dtype=torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
# Get quant config from template
template = LLMTemplate.get(model.config.model_type)
quant_config = template.get_config(scheme=quant_scheme, exclude_layers=exclude_layers)
# Quantize
quantizer = ModelQuantizer(quant_config)
model = quantizer.quantize_model(model)
model = quantizer.freeze(model)
# Export hf_format

Title: deepseek-ai/DeepSeek-V4-Flash - Hugging Face
URL: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
Content: ## Model Downloads

| Model | #Total Params | #Activated Params | Context Length | Precision | Download |
 ---  ---  --- |
| DeepSeek-V4-Flash-Base | 284B | 13B | 1M | FP8 Mixed | HuggingFace | ModelScope |
| DeepSeek-V4-Flash | 284B | 13B | 1M | FP4 + FP8 Mixed\ | HuggingFace | ModelScope |
| DeepSeek-V4-Pro-Base | 1.6T | 49B | 1M | FP8 Mixed | HuggingFace | ModelScope |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M | FP4 + FP8 Mixed\ | HuggingFace | ModelScope |

\FP4 + FP8 Mixed: MoE expert parameters use FP4 precision; most other parameters use FP8.

## Evaluation Results

### Base Model [...] | MCPAtlas Public (Pass@1) | 73.8 | 67.2 | 69.2 | 66.6 | 71.8 | 73.6 |
| Toolathlon (Pass@1) | 47.2 | 54.6 | 48.8 | 50.0 | 40.7 | 51.8 | [...] | GDPval-AA (Elo)  - | 1395  - | 1554 |
| Toolathlon (Pass@1) | 40.7 | 43.5 | 47.8 | 46.3 | 49.0 | 51.8 |

Title: Quantized Models for deepseek-ai/DeepSeek-V4-Flash – Hugging Face
URL: https://huggingface.co/models?other=base_model:quantized:deepseek-ai/DeepSeek-V4-Flash
Content: Clear all

Image 2 #### mlx-community/deepseek-ai-DeepSeek-V4-Flash-8bit Text Generation • 284B•Updated about 5 hours ago• 6Image 3 #### sgl-project/DeepSeek-V4-Flash-FP8 291B•Updated about 7 hours ago• 4Image 4 #### Thump604/DeepSeek-V4-Flash-MLX-Q2-mixed-gs128-affine Text Generation • 284B•Updated about 4 hours ago• 1Image 5 #### Thump604/DeepSeek-V4-Flash-MLX-Q3-mixed-gs128-affine Text Generation • 284B•Updated about 4 hours agoImage 6 #### mlx-community/deepseek-ai-DeepSeek-V4-Flash-6bit Text Generation • 284B•Updated about 4 hours agoImage 7 #### mlx-community/DeepSeek-V4-Flash-4bit Text Generation • 284B•Updated about 3 hours agoImage 8 #### mlx-community/deepseek-ai-DeepSeek-V4-Flash-4bit Text Generation • 284B•Updated about 3 hours ago

 System theme

Company [...] # Quantized Models for deepseek-ai/DeepSeek-V4-Flash – Hugging Face

Image 1: Hugging Face's logoHugging Face

   Models
   Datasets
   Spaces
   Buckets new
   Docs
   Enterprise
   Pricing
    
   
  

   Log In
   Sign Up

### Edit Models filters

   Main 
   Tasks 
   Libraries 
   Languages 
   Licenses 
   Other 1

Model Tree

 Reset

deepseek-ai/DeepSeek-V4-Flash

AdaptersFinetunesQuantizationsMerges

Apps 

llama.cppLM StudioJanDraw ThingsDiffusionBeeJoyFusionvLLMOllamaMLX LMDocker Model RunnerLemonadeSGLangUnslothPi

Inference Providers 

Select all

GroqNovitaCerebrasSambaNovaNscalefalHyperbolicTogether AIFireworksFeatherless AIZaiReplicateCohereScalewayPublic AIOVHcloud AI EndpointsHF Inference APIWaveSpeed

Misc [...] Misc 

Inference Endpointstext-generation-inferenceEval Results (legacy)text-embeddings-inference4-bit precisionMergecustom_code8-bit precisionMixture of ExpertsCarbon EmissionsEval Results

 Apply filters

# Models

7

Full-text search Inference Available

 Edit filters

 Sort: Trending

Active filters: deepseek-ai/DeepSeek-V4-Flash

Clear all

Title: config.json · deepseek-ai/DeepSeek-V4-Flash at main
URL: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
Content: |  | "num\_experts\_per\_tok": 6, |
|  | "num\_hidden\_layers": 43, |
|  | "num\_hash\_layers": 3, |
|  | "num\_key\_value\_heads": 1, |
|  | "num\_nextn\_predict\_layers": 1, |
|  | "o\_groups": 8, |
|  | "o\_lora\_rank": 1024, |
|  | "q\_lora\_rank": 1024, |
|  | "qk\_rope\_head\_dim": 64, |
|  | "quantization\_config": { |
|  | "activation\_scheme": "dynamic", |
|  | "fmt": "e4m3", |
|  | "quant\_method": "fp8", |
|  | "scale\_fmt": "ue8m0", |
|  | "weight\_block\_size": [ |
|  | 128, |
|  | 128 |
|  | ] |
|  | }, |
|  | "rms\_norm\_eps": 1e-06, |
|  | "rope\_scaling": { |
|  | "beta\_fast": 32, |
|  | "beta\_slow": 1, |
|  | "factor": 16, |
|  | "original\_max\_position\_embeddings": 65536, |
|  | "type": "yarn" |
|  | }, |
|  | "rope\_theta": 10000, | [...] Hugging Face's logo

# deepseek-ai / DeepSeek-V4-Flash like 247 Follow DeepSeek 126k

GeeeekExplorer's picture [...] |  |  |
 --- |
|  | { |
|  | "architectures": [ |
|  | "DeepseekV4ForCausalLM" |
|  | ], |
|  | "attention\_bias": false, |
|  | "attention\_dropout": 0.0, |
|  | "bos\_token\_id": 0, |
|  | "eos\_token\_id": 1, |
|  | "hc\_eps": 1e-06, |
|  | "hc\_mult": 4, |
|  | "hc\_sinkhorn\_iters": 20, |
|  | "head\_dim": 512, |
|  | "hidden\_act": "silu", |
|  | "hidden\_size": 4096, |
|  | "index\_head\_dim": 128, |
|  | "index\_n\_heads": 64, |
|  | "index\_topk": 512, |
|  | "initializer\_range": 0.02, |
|  | "max\_position\_embeddings": 1048576, |
|  | "model\_type": "deepseek\_v4", |
|  | "moe\_intermediate\_size": 2048, |
|  | "n\_routed\_experts": 256, |
|  | "n\_shared\_experts": 1, |
|  | "norm\_topk\_prob": true, |
|  | "num\_attention\_heads": 64, |
|  | "num\_experts\_per\_tok": 6, |

Title: Deepseek V4 released - DGX Spark / GB10
URL: https://forums.developer.nvidia.com/t/deepseek-v4-released/367696
Content: Image 13

azampatti

15h

I was just reading that too

Image 14: image image 1079×516 51.7 KB

And it’s an FP4+FP8 mixed one with over 130GB in size. We will have to have an INT4 or something like that to barely aspire to maybe running it

1 Reply

 1  ​ 

 ​ 

## post by adg1 15 hours ago

Image 15

adg1

1

15h

Image 16 eparin82:



> deepseek-ai/DeepSeek-V4-Flash · Hugging Face

All, this is huge. Make sure to read the technical report at  for a glimpse of the innovation behind this series of frontier level open-weight model.

From an operational point of view, DeepSeek-V4-Flash is the one in the series we will be tinkering with pretty soon, with 284B parameters (13B activated).

 1  ​ 

 ​ 

## post by JW2026 15 hours ago

## post by joshua.dale.warner 14 hours ago [...] Image 10

azampatti

16h

That’s the one I was referring to, Looks like a smaller model rather than a quantization. That one should run in a dual spark, but if all of those are active parameters, it’ll be slow… :) I would definitely love an FP8 version of that one though :)

 3  ​ 

 ​ 

## post by mclenithan 16 hours ago

Image 11

mclenithan

16h

Looks like the architecture is unique… not something to easily toss into vLLM with the current versions, we’ll have to figure this out. Super excited to see where this goes, my quad gb10 cluster is stoked.

1 Reply

 4  ​ 

 ​ 

## post by whpthomas 15 hours ago

Image 12

whpthomas

15h

DeepSeek-V4-Flash is 284B parameters A13B activated – So no for single spark `:(`

 2  ​ 

 ​ 

## post by azampatti 15 hours ago

Image 13

azampatti

15h [...] 3  ​ 

 ​ 

1.5k views 46 likes 5 links 22 users 

Image 3: Azampatti4

Image 42

Image 5: Patrick M2

Image 62

Image 7: grindstone

 read  4 min 

## post by azampatti 16 hours ago

Image 8

azampatti

16h

deepseek-ai/DeepSeek-V4-Flash This one looks like we might be able to run an FP8 quant in a single node…Maybe?

Interesting indeed!

 2  ​ 

 ​ 

## post by eparin82 16 hours ago

Image 9

eparin82

16h

Deepseek made a quantized version.

huggingface.co
### deepseek-ai/DeepSeek-V4-Flash · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.

3 Replies

 1  ​ 

 ​ 

## post by azampatti 16 hours ago

Image 10

azampatti

16h
```

---

## Assistant (Build · DeepSeek V4 Pro · 153.8s)

_Thinking:_

Now I have a much clearer picture. Let me summarize what I've found:

## DeepSeek V4 Pro / Flash Architecture

### Key differences from V2/V3:
1. **Architecture**: `"DeepseekV4ForCausalLM"` - a new architecture, NOT `DeepseekV2ForCausalLM` or `DeepseekV3ForCausalLM`
2. **model_type**: `"deepseek_v4"` - a new model type
3. **New features**:
   - Hybrid attention: CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention)
   - mHC (Manifold-Constrained Hyper-Connections)
   - Indexer with `index_head_dim`, `index_n_heads`, `index_topk`
   - `compress_rope_theta`, `compress_ratios` - for long context compression
   - `hc_eps`, `hc_mult`, `hc_sinkhorn_iters` - hyper-connection parameters
   - `scoring_func`, `swiglu_limit` - MoE scoring parameters
   - `sliding_window` - sliding window attention
4. **Quantization**: FP4 + FP8 Mixed
   - `quantization_config.quant_method` = `"fp8"` in the config.json
   - But FP4 is used for MoE expert parameters

### V4 Flash specifics:
- 284B total, 13B activated
- 43 layers (vs 61 for Pro)
- hidden_size: 4096 (vs 7168 for Pro)
- moe_intermediate_size: 2048
- n_routed_experts: 256
- head_dim: 512
- num_attention_heads: 64

### What vLLM currently has:
- `deepseek_v2.py` - supports V2 and V3 architectures
- The V4 architecture is different enough to need a new model file

Now, the user's problem: "some of the model parameters are quantized in mxfp4 which current plugin does not support"

Wait - the config.json says `quant_method: "fp8"`. So the model checkpoint itself declares FP8 quantization. But the FP4 part - how is it declared?

Let me look at the actual weight files. The config says FP8 but the model description says "FP4 + FP8 Mixed". This means:
- Most weights are FP8 (which vLLM already supports)
- MoE expert weights are FP4 (MXFP4 format)
- There may be additional quantization config files or weight naming conventions

Looking at the search results, one result says: "At deployment, V4 relies heavily on FP4 (MXFP4) quantization." And the Flash model page says the FP4+FP8 mixed version uses FP4 for MoE expert parameters.

So the issue is likely that:
1. The model architecture is new (V4) and not yet in vLLM
2. The mixed FP4+FP8 quantization requires special handling - specifically, some layers use FP4 (MXFP4) weights while others use FP8
3. The user's "NUMA plugin" may not be the right term - they probably mean the model support in general

Let me now look at how the V4 config differs from V3 to understand what model changes are needed. Also, I should check if anyone has started work on V4 support in vLLM.

Let me also look at the actual safetensors weight structure to understand the FP4 format better.

Actually, let me also look at the `deepseek_v2.py` more closely to understand exactly what pieces need to change for V4 support, and check the existing quantization auto-detection path.

Let me now write up a comprehensive analysis and plan for the user.

Let me look at the V4 model config more closely and check the existing deepseek model and quantization detection paths.

**Tool: tavily_tavily_extract**

**Input:**
```json
{
  "urls": [
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json",
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json"
  ],
  "format": "text"
}
```

**Output:**
```
Detailed Results:

Title: config.json · deepseek-ai/DeepSeek-V4-Flash at main - Hugging Face
URL: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
Content: undefined
Raw Content: Hugging Face's logo
deepseek-ai / DeepSeek-V4-Flash like 247 Follow DeepSeek 126k

GeeeekExplorer's picture
|  |  |
 --- |
|  | { |
|  | "architectures": [ |
|  | "DeepseekV4ForCausalLM" |
|  | ], |
|  | "attention\_bias": false, |
|  | "attention\_dropout": 0.0, |
|  | "bos\_token\_id": 0, |
|  | "eos\_token\_id": 1, |
|  | "hc\_eps": 1e-06, |
|  | "hc\_mult": 4, |
|  | "hc\_sinkhorn\_iters": 20, |
|  | "head\_dim": 512, |
|  | "hidden\_act": "silu", |
|  | "hidden\_size": 4096, |
|  | "index\_head\_dim": 128, |
|  | "index\_n\_heads": 64, |
|  | "index\_topk": 512, |
|  | "initializer\_range": 0.02, |
|  | "max\_position\_embeddings": 1048576, |
|  | "model\_type": "deepseek\_v4", |
|  | "moe\_intermediate\_size": 2048, |
|  | "n\_routed\_experts": 256, |
|  | "n\_shared\_experts": 1, |
|  | "norm\_topk\_prob": true, |
|  | "num\_attention\_heads": 64, |
|  | "num\_experts\_per\_tok": 6, |
|  | "num\_hidden\_layers": 43, |
|  | "num\_hash\_layers": 3, |
|  | "num\_key\_value\_heads": 1, |
|  | "num\_nextn\_predict\_layers": 1, |
|  | "o\_groups": 8, |
|  | "o\_lora\_rank": 1024, |
|  | "q\_lora\_rank": 1024, |
|  | "qk\_rope\_head\_dim": 64, |
|  | "quantization\_config": { |
|  | "activation\_scheme": "dynamic", |
|  | "fmt": "e4m3", |
|  | "quant\_method": "fp8", |
|  | "scale\_fmt": "ue8m0", |
|  | "weight\_block\_size": [ |
|  | 128, |
|  | 128 |
|  | ] |
|  | }, |
|  | "rms\_norm\_eps": 1e-06, |
|  | "rope\_scaling": { |
|  | "beta\_fast": 32, |
|  | "beta\_slow": 1, |
|  | "factor": 16, |
|  | "original\_max\_position\_embeddings": 65536, |
|  | "type": "yarn" |
|  | }, |
|  | "rope\_theta": 10000, |
|  | "routed\_scaling\_factor": 1.5, |
|  | "scoring\_func": "sqrtsoftplus", |
|  | "sliding\_window": 128, |
|  | "swiglu\_limit": 10.0, |
|  | "tie\_word\_embeddings": false, |
|  | "topk\_method": "noaux\_tc", |
|  | "torch\_dtype": "bfloat16", |
|  | "transformers\_version": "4.57.1", |
|  | "use\_cache": true, |
|  | "vocab\_size": 129280, |
|  | "compress\_rope\_theta": 160000, |
|  | "compress\_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0] |
|  | } |
|  |  |

Title: config.json · deepseek-ai/DeepSeek-V4-Pro at main
URL: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json
Content: undefined
Raw Content: Hugging Face's logo
deepseek-ai / DeepSeek-V4-Pro like 1.09k Follow DeepSeek 126k

GeeeekExplorer's picture
|  |  |
 --- |
|  | { |
|  | "architectures": [ |
|  | "DeepseekV4ForCausalLM" |
|  | ], |
|  | "attention\_bias": false, |
|  | "attention\_dropout": 0.0, |
|  | "bos\_token\_id": 0, |
|  | "eos\_token\_id": 1, |
|  | "hc\_eps": 1e-06, |
|  | "hc\_mult": 4, |
|  | "hc\_sinkhorn\_iters": 20, |
|  | "head\_dim": 512, |
|  | "hidden\_act": "silu", |
|  | "hidden\_size": 7168, |
|  | "index\_head\_dim": 128, |
|  | "index\_n\_heads": 64, |
|  | "index\_topk": 1024, |
|  | "initializer\_range": 0.02, |
|  | "max\_position\_embeddings": 1048576, |
|  | "model\_type": "deepseek\_v4", |
|  | "moe\_intermediate\_size": 3072, |
|  | "n\_routed\_experts": 384, |
|  | "n\_shared\_experts": 1, |
|  | "norm\_topk\_prob": true, |
|  | "num\_attention\_heads": 128, |
|  | "num\_experts\_per\_tok": 6, |
|  | "num\_hidden\_layers": 61, |
|  | "num\_hash\_layers": 3, |
|  | "num\_key\_value\_heads": 1, |
|  | "num\_nextn\_predict\_layers": 1, |
|  | "o\_groups": 16, |
|  | "o\_lora\_rank": 1024, |
|  | "q\_lora\_rank": 1536, |
|  | "qk\_rope\_head\_dim": 64, |
|  | "quantization\_config": { |
|  | "activation\_scheme": "dynamic", |
|  | "fmt": "e4m3", |
|  | "quant\_method": "fp8", |
|  | "scale\_fmt": "ue8m0", |
|  | "weight\_block\_size": [ |
|  | 128, |
|  | 128 |
|  | ] |
|  | }, |
|  | "rms\_norm\_eps": 1e-06, |
|  | "rope\_scaling": { |
|  | "beta\_fast": 32, |
|  | "beta\_slow": 1, |
|  | "factor": 16, |
|  | "original\_max\_position\_embeddings": 65536, |
|  | "type": "yarn" |
|  | }, |
|  | "rope\_theta": 10000, |
|  | "routed\_scaling\_factor": 2.5, |
|  | "scoring\_func": "sqrtsoftplus", |
|  | "sliding\_window": 128, |
|  | "swiglu\_limit": 10.0, |
|  | "tie\_word\_embeddings": false, |
|  | "topk\_method": "noaux\_tc", |
|  | "torch\_dtype": "bfloat16", |
|  | "transformers\_version": "4.57.1", |
|  | "use\_cache": true, |
|  | "vocab\_size": 129280, |
|  | "compress\_rope\_theta": 160000, |
|  | "compress\_ratios": [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0] |
|  | } |
|  |  |
```

**Tool: task**

**Input:**
```json
{
  "description": "Read key deepseek and quant files",
  "subagent_type": "explore",
  "prompt": "Very thorough exploration. Read the following files completely and return their full contents:\n\n1. /Users/ross/Documents/project/Lvllm/vllm/model_executor/models/deepseek_v2.py - focus on:\n   - The class hierarchy (what does DeepseekV2ForCausalLM inherit from?)\n   - How model_type is registered (look for \"deepseek_v2\" or \"deepseek_v3\" strings)\n   - The Indexer class and its is_fp4_ckpt handling\n   - How attention types are selected (MLA vs standard)\n   - How the model handles different configs from config.json\n   - The load_weights() method\n\n2. /Users/ross/Documents/project/Lvllm/vllm/config/model.py - focus on:\n   - The _verify_quantization() method \n   - How quantization_config is read from HF config\n   - The override_quantization_method flow\n\n3. /Users/ross/Documents/project/Lvllm/vllm/model_executor/models/registry.py - how models are registered, how model_type maps to model class\n\n4. /Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/modelopt.py - focus on:\n   - ModelOptMixedPrecisionConfig class\n   - How mixed precision (FP8 + FP4) is handled\n   - How different layers get different quantization\n\nReturn the complete content of each file or relevant sections."
}
```

**Output:**
```
task_id: ses_2339b0d45ffegjikd8jEpNcy2X (for resuming to continue this task if needed)

<task_result>
Now I have all files read completely. Here is a comprehensive summary focusing on the specific areas you asked about:

---

## 1. `/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/deepseek_v2.py`

### Class Hierarchy of `DeepseekV2ForCausalLM`

```python
class DeepseekV2ForCausalLM(
    nn.Module,
    SupportsPP,           # Supports pipeline parallelism
    DeepseekV2MixtureOfExperts,  # MoE support
    SupportsLoRA,         # LoRA support
    SupportsEagle,        # Eagle speculative decoding
    SupportsEagle3,       # Eagle3 speculative decoding
):
```

It inherits from:
- `nn.Module` (PyTorch base)
- `SupportsPP` (pipeline parallelism interface, from `.interfaces`)
- `DeepseekV2MixtureOfExperts` (which itself inherits from `MixtureOfExperts`)
- `SupportsLoRA`, `SupportsEagle`, `SupportsEagle3` (mixins)

There are also alias subclasses at the bottom:
```python
class DeepseekForCausalLM(DeepseekV2ForCausalLM): pass
class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM): pass
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass
```

### How `model_type` is Registered (registry mapping)

The mapping is in `/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/registry.py`, lines 95-99:
```python
"DeepseekForCausalLM": ("deepseek_v2", "DeepseekForCausalLM"),
"DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
"DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
"DeepseekV32ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
# Also:
"GlmMoeDsaForCausalLM": ("deepseek_v2", "GlmMoeDsaForCausalLM"),
```

All DeepSeek variants map to the `deepseek_v2` module, but point to different class names. Note: `DeepseekV32ForCausalLM` maps to class `DeepseekV3ForCausalLM` (not `DeepseekV32`).

### The `Indexer` class and its `is_fp4_ckpt` handling

The `Indexer` class (lines 613-759) is the sparse attention indexer for DeepSeek V3.2 models:

```python
class Indexer(nn.Module):
    def __init__(self, ..., quant_config, ...):
        self.is_fp4_ckpt = (
            self.quant_config is not None
            and self.quant_config.get_name() == "modelopt_fp4"
        )
```

**When `is_fp4_ckpt` is True:**
- It creates a **fused** `wk_weights_proj` linear layer (MergedColumnParallelLinear) that combines `wk` (key projection) and `weights_proj` (score weights) into a single GEMM. **Crucially, this fused layer gets `quant_config=None`** (unquantized), because `weights_proj` is not quantized.
- The `wk` projection may be upcast from the default quant; the comment says "experiments show fusion is always faster unless WK proj is in FP4, which is not the case for all known quants."

**When `is_fp4_ckpt` is False:**
- It creates separate `wk` (ReplicatedLinear with quant_config) and `weights_proj` (ReplicatedLinear with `quant_config=None`) layers.

In the `forward()` method:
- If `is_fp4_ckpt`: runs the fused GEMM then splits the output into `k` and `weights`.
- Otherwise: runs separate GEMMs.

### How attention types are selected (MLA vs standard)

In `DeepseekV2DecoderLayer.__init__()` (lines 1060-1077):
```python
use_mha = config.model_type == "deepseek" or all(
    dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
)
# ...
if use_mha:
    attn_cls = DeepseekAttention    # Standard MHA (v1)
elif model_config.use_mla:
    attn_cls = DeepseekV2MLAAttention  # MLA with V3.2 sparse indexer support
else:
    attn_cls = DeepseekV2Attention     # MLA without sparse indexer
```

So attention is selected based on:
1. If `model_type == "deepseek"` or all MLA-specific dims are 0 -> use standard MHA (`DeepseekAttention`)
2. If `model_config.use_mla` is True -> use MLA with V3.2 support (`DeepseekV2MLAAttention`)
3. Otherwise -> use basic MLA (`DeepseekV2Attention`)

### How the model handles different configs from `config.json`

- `model_type`: Used to determine if MHA is needed (`"deepseek"` vs `"deepseek_v2"`/`"deepseek_v3"`)
- `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `kv_lora_rank`: All checked via `getattr(config, ..., 0)` with defaults of 0; when all are 0, MHA is forced
- `q_lora_rank`: Checked via `hasattr(config, "q_lora_rank")`; if present and non-None, fused QKV-A projections are used
- `index_topk`: In `DeepseekV2MLAAttention` and `DeepseekV2Model`, `hasattr(config, "index_topk")` determines if it's V3.2 (with sparse attention indexer)
- `num_nextn_predict_layers`: For MTP (speculative decoding) layers in `get_spec_layer_idx_from_weight_name()`
- `moe_layer_freq`, `first_k_dense_replace`, `n_routed_experts`, `n_shared_experts`: Control MoE vs dense MLP placement
- `routed_scaling_factor`, `norm_topk_prob`, `n_group`, `topk_group`, `scoring_func`: MoE routing parameters
- `rope_parameters`: Determines RoPE type (yarn vs llama scaling)
- `llama_4_scaling`: Optional per-token scaling from Llama 4 models
- `topk_method == "noaux_tc"`: Controls expert score correction bias

### `load_weights()` method (lines 1454-1666)

Key aspects:
1. Defines `stacked_params_mapping` for gate/up projection fusion, MLA param mapping, MHA param mapping
2. For `is_fp4_ckpt`: extends `stacked_params_mapping` with indexer fused mapping (`wk_weights_proj`)
3. Uses `expert_params_mapping` from `SharedFusedMoE.make_expert_params_mapping()` for expert weights
4. Weight loading flow:
   - Skips rotary_emb.inv_freq and spec decode layer weights
   - Tries `stacked_params_mapping` first (for fused projections like `gate_up_proj`, `fused_qkv_a_proj`, `qkv_proj`, `wk_weights_proj`)
   - Then tries `expert_params_mapping` for MoE expert weights
   - Special handling for AITER `fusion_moe_shared_experts` (splits shared expert tensors and routes to expert slots)
   - Falls back to direct `default_weight_loader` for unmatched weights
   - Remaps FP8 kv-scale names via `maybe_remap_kv_scale_name()`

---

## 2. `/Users/ross/Documents/project/Lvllm/vllm/config/model.py`

### `_verify_quantization()` method (lines 929-1022)

```python
def _verify_quantization(self) -> None:
    supported_quantization = me_quant.QUANTIZATION_METHODS
    if self.quantization is not None:
        self.quantization = cast(me_quant.QuantizationMethods, self.quantization)

    # Parse quantization method from the HF model config, if available.
    quant_cfg = self.model_arch_config.quantization_config

    if quant_cfg is not None:
        quant_method = quant_cfg["quant_method"]
        # ...
```

Key flow:
1. Gets the `quantization_config` dict from the model architecture config (parsed from HF's `config.json`)
2. If `quant_cfg` exists, extracts `quant_method` from it
3. Builds a priority-ordered list of quantization backends, with "override" backends (those that implement `override_quantization_method()`) checked last
4. **Override backends in order of preference**: `gptq_marlin`, `awq_marlin`, `inc`, `moe_wna16`, `modelopt`, `modelopt_fp4`, `modelopt_mxfp8`, `modelopt_mixed`, `mxfp4`, `cpu_awq`, `gguf`
5. For each quantization method, calls `method.override_quantization_method(quant_cfg, self.quantization)` to see if it should take over
6. If an override returns non-None, that becomes the resolved quantization method
7. Validates that if user specified `--quantization`, it matches what was auto-detected
8. Calls `current_platform.verify_quantization()` for platform-specific checks
9. Checks for deprecated quantization methods

### How `quantization_config` is read from HF config

The quantization_config is read in `ModelConfig.__post_init__()` via:
```python
hf_config = get_config(...)
self.hf_config = hf_config
# ...
self.model_arch_config = self.get_model_arch_config()
```
Then in `_verify_quantization()`:
```python
quant_cfg = self.model_arch_config.quantization_config
```

The `model_arch_config` is created by:
```python
def get_model_arch_config(self):
    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
        self.hf_config.model_type, ModelArchConfigConvertorBase
    )
    convertor = convertor_cls(self.hf_config, self.hf_text_config)
    return convertor.convert()
```
This convertor extracts `quantization_config` from the HF config (which may be in `config.json` or `hf_quant_config.json`).

### The `override_quantization_method` flow

The overrides list defines the order:
```python
overrides = [
    "gptq_marlin", "awq_marlin", "inc", "moe_wna16",
    "modelopt", "modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed",
    "mxfp4", "cpu_awq", "gguf",
]
```

For each override, the code:
1. Gets the quantization config class via `me_quant.get_quantization_config(name)`
2. Calls `method.override_quantization_method(quant_cfg, self.quantization)`
3. If it returns a string (the effective quant method name), that overrides both `quant_method` and `self.quantization`
4. The first matching override wins

For example, `ModelOptFp8Config.override_quantization_method()` extracts the `quant_algo` from the config and returns `"modelopt"` if it's `"FP8"`. Similarly, `ModelOptNvFp4Config` returns `"modelopt_fp4"` for NVFP4/FP4.

---

## 3. `/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/registry.py`

### How models are registered

The registry uses a `_ModelRegistry` dataclass (line 893):
```python
@dataclass
class _ModelRegistry:
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)
```

It supports two registration types:
1. **Eager** (`_RegisteredModel`): Already imported, stores the class directly
2. **Lazy** (`_LazyRegisteredModel`): Stores `module_name` and `class_name` as strings, imports on demand

### `model_type` to model class mapping

The mapping is defined by dictionaries like `_TEXT_GENERATION_MODELS`, `_EMBEDDING_MODELS`, `_MULTIMODAL_MODELS`, etc., all merged into `_VLLM_MODELS` (line 633).

Each entry maps `HF_ARCH_NAME -> (module_relname, vllm_class_name)`:
```python
"DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
```

### Final `ModelRegistry` singleton (line 1281)

```python
ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm.model_executor.models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
    }
)
```

All models are registered lazily. When `resolve_model_cls()` or `inspect_model_cls()` is called:
1. It checks if the architecture is in `self.models`
2. If `model_impl == "transformers"`, tries the transformers backend
3. Falls back to transformers backend if no vllm model found and `model_impl == "auto"`
4. Uses `_normalize_arch()` to handle model architecture suffixes (e.g., `ForCausalLM`)
5. Finally calls `_raise_for_unsupported()` if nothing matches, with helpful messages for previously-supported and OOT-supported models

The `register_model()` method allows external models to be registered at runtime with either a class or a string `"module:class"`.

---

## 4. `/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/modelopt.py`

### `ModelOptMixedPrecisionConfig` class (lines 2005-2187)

```python
class ModelOptMixedPrecisionConfig(ModelOptQuantConfigBase):
    """Config class for ModelOpt MIXED_PRECISION.

    Supports checkpoints where different layers use different quantization
    algorithms (e.g., FP8 for dense layers and NVFP4 for MoE experts).
    """
```

Key internals:
- `quantized_layers`: A `dict[str, dict[str, Any]]` mapping layer names to their per-layer quantization config
- `fp8_config`: A `ModelOptFp8Config` instance used for FP8-quantized layers
- `nvfp4_config`: A `ModelOptNvFp4Config` instance used for NVFP4-quantized layers

### How mixed precision (FP8 + FP4) is handled

**`get_name()`** returns `"modelopt_mixed"`.

**`override_quantization_method()`** checks if `quant_algo == "MIXED_PRECISION"` and returns `"modelopt_mixed"`.

**`_from_config()`** (lines 2048-2100):
1. Extracts `quantized_layers` from the quantization config
2. Determines `group_size` by scanning NVFP4 entries (defaults to 16)
3. Creates child `fp8_config` and `nvfp4_config` instances (with empty exclude_modules since per-layer control supersedes global exclusion)

**`_resolve_quant_algo(prefix)`** (lines 2102-2141) resolves which quant algorithm applies to a given vLLM layer prefix using three strategies:
1. **Direct lookup**: prefix in `quantized_layers`
2. **Packed/fused layer lookup**: unfuses via `packed_modules_mapping` and checks each shard
3. **Prefix-based lookup**: for FusedMoE parent modules, checks if any child key starts with `prefix + "."`

**`get_quant_method(layer, prefix)`** (lines 2143-2182):
- For `Attention`: returns KV-cache method if specified
- For excluded layers: returns `UnquantizedLinearMethod()`
- For `LinearBase`: delegates to the appropriate LinearMethod (`ModelOptFp8LinearMethod` or `ModelOptNvFp4LinearMethod`)
- For `FusedMoE`: delegates to the appropriate MoE method (`ModelOptFp8MoEMethod` or `ModelOptNvFp4FusedMoE`)

### How different layers get different quantization

The per-layer quantization is driven by the `quantized_layers` dict from the checkpoint's quantization config. For example, a config might specify:
- `"model.layers.*.self_attn.*"` -> `quant_algo: "FP8"`
- `"model.layers.*.mlp.experts.*"` -> `quant_algo: "NVFP4"`

This is then resolved per-layer during model initialization when `get_quant_method()` is called for each linear/MoE layer by vLLM's layer creation logic. The `_resolve_quant_algo()` method handles both exact layer names and wildcard-style lookups using `packed_modules_mapping` and prefix matching.

**`apply_vllm_mapper()`** (line 2184): Also remaps the `quantized_layers` dict through the vLLM weights mapper (which handles any name remapping needed by the vLLM implementation).

### Other classes in this file

- **`ModelOptFp8Config`**: For FP8 quantization (quant_algo: FP8, FP8_PER_CHANNEL_PER_TOKEN, FP8_PB_WO). Returns `"modelopt"` as name.
- **`ModelOptNvFp4Config`**: For NVFP4 (FP4) quantization. Returns `"modelopt_fp4"` as name.
- **`ModelOptMxFp8Config`**: For MXFP8 block-scaled FP8. Returns `"modelopt_mxfp8"` as name.
- **`ModelOptMixedPrecisionConfig`**: For mixed precision (FP8 + NVFP4). Returns `"modelopt_mixed"` as name.

Each has corresponding LinearMethod, FusedMoEMethod, and KVCacheMethod classes registered as class attributes.
</task_result>
```

---

## Assistant (Build · DeepSeek V4 Pro · 56.3s)

_Thinking:_

Now I have a comprehensive understanding. Let me analyze the situation:

## Key Findings

### DeepSeek V4 Architecture
The config.json shows:
1. `"architectures": ["DeepseekV4ForCausalLM"]`
2. `"model_type": "deepseek_v4"`
3. `"quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [128, 128], "activation_scheme": "dynamic"}`
4. New config keys not in V2/V3: `hc_eps`, `hc_mult`, `hc_sinkhorn_iters`, `head_dim`, `index_head_dim`, `index_n_heads`, `index_topk`, `compress_rope_theta`, `compress_ratios`, `scoring_func`, `swiglu_limit`, `sliding_window`, `num_hash_layers`

### What vLLM has vs what's needed

The current `deepseek_v2.py` supports:
- `model_type: "deepseek_v2"` and `model_type: "deepseek_v3"`
- Standard MHA and MLA attention
- Sparse attention indexer (V3.2)
- `modelopt_fp4` (NVFP4) quantization via `is_fp4_ckpt` check
- FP8 quantization via modelopt

What's missing for V4:
1. **Model class not registered**: `DeepseekV4ForCausalLM` with `model_type: "deepseek_v4"` is not in the registry
2. **New architecture features**: Heavily Compressed Attention (HCA), mHC connections, Compressed Sparse Attention (CSA)
3. **Mixed FP4+FP8**: The quantization config says `quant_method: "fp8"` but the actual weights include FP4 (MXFP4) for MoE experts

### The Key Issue: Mixed FP4+FP8 Quantization

The config.json says `quantization_config.quant_method = "fp8"` - this is what DeepSeek's model declares. But according to the model description: "FP4 + FP8 Mixed: MoE expert parameters use FP4 precision; most other parameters use FP8."

So the quantization config in config.json is misleading - it says `quant_method: "fp8"` but the actual weight files for MoE experts are stored in FP4 format.

This means:
1. vLLM will detect `quant_method: "fp8"` from the config.json
2. It will try to use FP8 quantization for all layers
3. But when it reaches MoE expert weights, they're in FP4 format → loading fails

OR the model uses a different quantization config file (`hf_quant_config.json`) that's separate from `config.json`.

Let me check what the actual hf_quant_config.json or the actual weight formats look like.

Actually, looking more carefully at the quantization config:
```json
"quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128]
}
```

This `fmt: "e4m3"` + `scale_fmt: "ue8m0"` pattern is similar to what MXFP8 uses (E4M3 with E8M0 scales, block size 128). But the block size 128,128 matches FP8 block quantization, not MXFP4 (which uses block size 32).

So the quantization config in config.json represents the FP8 portion. The FP4 expert weights must be specified elsewhere, possibly in a separate `hf_quant_config.json` or in the quantized layers configuration.

Wait - let me re-read. The model is "FP4 + FP8 Mixed" where MoE experts are in FP4. This could be implemented in several ways:

1. The `config.json` has `quantization_config` for the overall model (FP8), and there's a separate `hf_quant_config.json` with per-layer info specifying which layers are FP4
2. The weight files themselves encode the format (different tensor shapes/dtypes for FP4 vs FP8 weights)
3. The model uses NVFP4 for MoE experts, which would need the `ModelOptMixedPrecisionConfig` with per-layer config

Actually, looking at the DeepSeek V4 technical report description: "At deployment, V4 relies heavily on FP4 (MXFP4) quantization." This suggests MXFP4, not NVFP4.

But vLLM's quantization ecosystem has TWO separate FP4 formats:
- **MXFP4**: OCP MX format, block size 32, E8M0 scales, supported via `Mxfp4Config` (`mxfp4.py`)
- **NVFP4**: NVIDIA format, block size 16, float8_e4m3fn scales, supported via `ModelOptNvFp4Config` (`modelopt.py`)

The `ModelOptMixedPrecisionConfig` only supports NVFP4, not MXFP4.

So the plan needs to be:

### Overall Approach

**Step 1: Register the model architecture**
Add `DeepseekV4ForCausalLM` to the registry, mapping to a new `deepseek_v4.py` module (or extending `deepseek_v2.py`).

**Step 2: Implement the model architecture**
The V4 model needs:
- CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention) hybrid
- mHC (Manifold-Constrained Hyper-Connections)
- New attention mechanism (likely different from MLA)
- Indexer with sparse attention
- New configuration options

**Step 3: Handle the mixed FP4+FP8 quantization**
Need to understand how the FP4 weights are actually structured. Options:
- If it's NVFP4: use `ModelOptMixedPrecisionConfig`
- If it's MXFP4: need to extend `ModelOptMixedPrecisionConfig` or create a custom solution
- If the quantization config just says fp8 but weights are fp4: need custom weight loader

**Step 4: Memory optimization for 16GB VRAM**
DeepSeek V4 Flash is 284B params. Even with FP4/FP8 quantization, the full model is ~80-130GB. With 16GB VRAM this requires:
- Aggressive CPU offloading
- Pipeline parallelism
- Layer-by-layer loading
- KV cache offloading to CPU

Let me now explore more about:
1. Whether there's already work on V4 support in vLLM upstream
2. What the actual weight files look like (check hf_quant_config.json)
3. The memory planning / offloading mechanisms

Actually, I think I have enough info now to give the user a comprehensive analysis. Let me also check what the actual quantized weight format looks like by looking at the model files.

Let me also check for any existing FP4+FP8 mixed support patterns.

Actually, let me focus on what the user specifically asked: "how to add DeepSeek V4 Pro support to this vllm NUMA plugin." They mentioned mxfp4 doesn't work, but the research shows mxfp4 IS partially supported (for MoE layers only, not for dense layers).

The core issue is likely:
1. The model architecture is new and not registered
2. The quantization is mixed (FP8 for dense, FP4 for MoE), which needs `ModelOptMixedPrecisionConfig` but with MXFP4 support added

Let me present my analysis to the user now and outline the implementation plan.

Wait, actually - the user said "the NUMA plugin" which is a specific thing (numactl --interleave=all). Maybe they literally mean they need to extend the NUMA interleaving to work with the new model's memory patterns? That seems unlikely. More likely they mean "the vLLM plugin for model support."

Let me give them the analysis but also ask clarifying questions.

Now I have a thorough understanding of the codebase. Let me also check if there's any upstream work on DeepSeek V4 and look at the weight file structure.

**Tool: tavily_tavily_search**

**Input:**
```json
{
  "query": "site:github.com/vllm-project/vllm deepseek v4 support pull request",
  "search_depth": "advanced"
}
```

**Output:**
```
Detailed Results:

Title: [Feature]: deepseek v4 support · Issue #40778 · vllm-project/vllm
URL: https://github.com/vllm-project/vllm/issues/40778
Content: devices:
            - driver: nvidia
              device_ids: ["0","1","2","3","4","5","6","7"]
              capabilities: [gpu]
undefined [...] # the same problem [...] - deepseek_v4
            - --served-model-name
            - ds-v4-pro
          image: vllm/vllm-openai:deepseekv4-cu129
          imagePullPolicy: IfNotPresent
          name: vllm-openai
          ports:
            - containerPort: 80
              protocol: TCP
          resources:
            limits:
              nvidia.com/gpu: "8"
            requests:
              nvidia.com/gpu: "8"
          env:
            - name: VLLM_USE_MODELSCOPE
              value: "true"
            - name: VLLM_ENGINE_READY_TIMEOUT_S
              value: "3600"
```

Title: [Feature]: Support DeepSeek V4 flash on SM120 with Triton fallback · Issue #40928 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/40928
Content: ## 🚀 The feature, motivation and pitch

I am trying to run DeepSeek V4 / DeepSeek-V4-Flash on NVIDIA SM120 GPUs. Currently, the DeepSeek V4 path depends on optimized kernels such as DeepGEMM and FlashMLA, but these kernels do not appear to support SM120 yet.

Because of this, DeepSeek V4 cannot run on SM120 even though the GPUs have enough memory and compute capability for the model. It would be very helpful if vLLM could support DeepSeek V4 on SM120, or provide a compatible execution path when DeepGEMM / FlashMLA are unavailable for this architecture.

SM120 GPUs are becoming available in workstation and server environments, so supporting this architecture would make DeepSeek V4 usable on newer NVIDIA hardware.

## Alternatives

The current alternatives seem to be: [...] ## Alternatives

The current alternatives seem to be:

I am not sure whether SM120 support is currently planned for DeepSeek V4 in vLLM.

## Additional context

Environment:

### Alternatives

No response

### Additional context

No response

### Before submitting a new issue...

## Metadata

## Metadata

### Assignees

### Labels

### Type

### Projects

### Milestone

### Relationships

### Development

## Issue actions

## Footer

### Footer navigation [...] ## Navigation Menu

# Search code, repositories, users, issues, pull requests...

# Provide feedback

We read every piece of feedback, and take your input very seriously.

# Saved searches

## Use saved searches to filter your results more quickly

To see all available qualifiers, see our documentation.

### Uh oh!

There was an error while loading. Please reload this page.

There was an error while loading. Please reload this page.

# [Feature]: Support DeepSeek V4 flash on SM120 with Triton fallback #40928

@bbbearxyz

## Description

@bbbearxyz

### 🚀 The feature, motivation and pitch

# Support DeepSeek V4 on SM120

## 🚀 The feature, motivation and pitch

Title: [Feature]: DeepSeek-V4 Flash sm_80 (A100/A800) support · Issue #40851 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/40851
Content: `vllm/vllm-openai:deepseekv4-cu129`
`DeepSeek-V4-Flash`

Simplified command:

### Before submitting a new issue...

## Metadata

## Metadata

### Assignees

### Labels

### Type

### Projects

### Milestone

### Relationships

### Development

## Issue actions

## Footer

### Footer navigation [...] `vllm/vllm-openai:deepseekv4-cu129`
`RuntimeError: Assertion error (.../deepgemm-src/csrc/apis/hyperconnection.hpp:56): Unsupported architecture`

This still happens with --enforce-eager, so it seems related to the DeepGEMM / HyperConnection kernels rather than torch.compile.

Are there any plans to support sm\_80 GPUs such as A100/A800 for DeepSeek-V4, possibly with a slower fallback path? A100/A800 GPUs are still widely used in many research and production clusters, so even a functional fallback would be very helpful.

### Alternatives

The current alternatives seem to be:

I am not sure whether sm\_80 support is planned or whether it should currently be considered unsupported for this model.

### Additional context

Environment:

`vllm/vllm-openai:deepseekv4-cu129`
`DeepSeek-V4-Flash` [...] ## Navigation Menu

# Search code, repositories, users, issues, pull requests...

# Provide feedback

We read every piece of feedback, and take your input very seriously.

# Saved searches

## Use saved searches to filter your results more quickly

To see all available qualifiers, see our documentation.

### Uh oh!

There was an error while loading. Please reload this page.

There was an error while loading. Please reload this page.

# [Feature]: DeepSeek-V4 Flash sm\_80 (A100/A800) support #40851

@xhd0728

## Description

@xhd0728

### 🚀 The feature, motivation and pitch

I am trying to run DeepSeek-V4-Flash on NVIDIA A800 GPUs, which are sm\_80 / Ampere GPUs. Currently, using `vllm/vllm-openai:deepseekv4-cu129`, the model fails during initialization with:

Title: [RFC]: DeepSeek-V4 Roadmap · Issue #8690 · vllm-project/vllm-ascend · GitHub
URL: https://github.com/vllm-project/vllm-ascend/issues/8690
Content: ## Navigation Menu

# Search code, repositories, users, issues, pull requests...

# Provide feedback

We read every piece of feedback, and take your input very seriously.

# Saved searches

## Use saved searches to filter your results more quickly

To see all available qualifiers, see our documentation.

# [RFC]: DeepSeek-V4 Roadmap #8690

@GDzhu01

## Description

@GDzhu01

### 📌 Current Status

As of releases/0.13.0, we have initial support for DeepSeek-V4-FLASH.  
For implementation details, please refer to #8648.

✨ On the horizon: full-fledged support for the DeepSeek-V4 model will land in releases 0.18.0 in next few days

### 🧭 Roadmap.

## Metadata

## Metadata

### Assignees

### Labels

### Type

### Projects

### Milestone

### Relationships

### Development

## Issue actions [...] ### Relationships

### Development

## Issue actions

## Footer

### Footer navigation

Title: 2026-04-24-deepseek-v4.md - GitHub
URL: https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md
Content: Raw

Copy raw file

Download raw file

You must be signed in to make or propose changes

More edit options

Outline

Edit and raw actions

| layout | post |

| title | DeepSeek V4 in vLLM: Efficient Long-context Attention |
| author | vLLM Team |
| image | /assets/figures/2026-04-dsv4/thumbnail.png |
| summary | A first-principles walkthrough of DeepSeek V4's long-context attention, and how we implemented it in vLLM. |
| tags | model-support |

We are excited to announce that vLLM now supports the DeepSeek V4 family of models (`deepseek-ai/DeepSeek-V4-Pro` and `deepseek-ai/DeepSeek-V4-Flash`). [...] This represents our initial release of model support, and further optimizations are actively underway. We hope the technical explanation that follows can help the open-source community understand both the attention mechanism itself and the rationale behind our current implementation decisions.

## Running DeepSeek V4 on vLLM

, which is substantially more memory-efficient than standard Multi-head Attention (MHA) or Multi-Query Attention (MQA), scaling to one million tokens remains difficult given the limited capacity of GPU memory.
   Attention computation cost: Computing attention over long contexts is expensive. Even with prior techniques such as DeepSeek Sparse Attention (DSA), the computation remains a significant bottleneck. [...] _sass  
         assets  
       .gitignore  
       404.html  
       Gemfile  
       README.md  
       _config.yml  
       favicon.ico  
       index.md  

## Breadcrumbs

1.   vllm-project.github.io
2.   /_posts

/
# 2026-04-24-deepseek-v4.md

Copy path

Blame More file actions

Blame More file actions

## Latest commit

Image 1: youkaichaoyoukaichao

Add reference link to related discussions (#198)

Open commit details success

Apr 24, 2026

94def92·Apr 24, 2026

## History

History

Open commit details

 · 22.6 KB

## Breadcrumbs

1.   vllm-project.github.io
2.   /_posts

/
# 2026-04-24-deepseek-v4.md

Top

## File metadata and controls

   Preview 
   Code 
   Blame 

executable file

·

293 lines (195 loc) · 22.6 KB

Raw

Copy raw file

Download raw file
```

**Tool: tavily_tavily_extract**

**Input:**
```json
{
  "urls": [
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main"
  ],
  "format": "text"
}
```

**Output:**
```
Detailed Results:

Title: deepseek-ai/DeepSeek-V4-Flash at main
URL: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main
Content: undefined
Raw Content: deepseek-ai/DeepSeek-V4-Flash at main
Image 1: Hugging Face's logoHugging Face
   Models
   Datasets
   Spaces
   Buckets new
   Docs
   Enterprise
   Pricing
    
   
  
   Log In
   Sign Up
Image 2
deepseek-ai
/
DeepSeek-V4-Flash
like 71
Follow
Image 3DeepSeek 126k
Text GenerationTransformersSafetensorsdeepseek_v48-bit precisionfp8
License:mit
Model cardFiles Files and versions xetCommunity 2
Deploy
Use this model
main
DeepSeek-V4-Flash
160 GB
⌘ K
⌘ K
 Image 4: msr2000's picture  Image 5  2 contributorsHistory:12 commits
Image 6: msr2000's picture
msr2000
Update benchmark figure
0cad8ee19 minutes ago
   assets Update benchmark figure19 minutes ago
   encoding Release DeepSeek-V4about 1 hour ago
   inference Update inference/generate.pyabout 1 hour ago
   .gitattributesSafe  1.6 kBRelease DeepSeek-V4about 1 hour ago
   DeepSeek_V4.pdf  4.48 MB xetUpdate technical report32 minutes ago
   LICENSESafe  1.08 kBRelease DeepSeek-V4about 1 hour ago
   README.md  13.8 kBUpdate technical report linkabout 1 hour ago
   config.json  1.72 kBUpload config.jsonabout 1 hour ago
   generation_config.json  170 BytesRelease DeepSeek-V4about 1 hour ago
   model-00001-of-00046.safetensors  1.06 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00002-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00003-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00004-of-00046.safetensors  3.6 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00005-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00006-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00007-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00008-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00009-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00010-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00011-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00012-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00013-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00014-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00015-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00016-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00017-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00018-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00019-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00020-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00021-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00022-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00023-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00024-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00025-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00026-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00027-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00028-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00029-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00030-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00031-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00032-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00033-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00034-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00035-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00036-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00037-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00038-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00039-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00040-of-00046.safetensors  3.59 GB xetAdd files using upload-large-folder toolabout 19 hours ago
   model-00041-of-00046.safetensors  3.57 GB xetAdd files using upload-large-folder toolabout 19 hours ago
    Load more files
Total file size
```

---

## Assistant (Build · DeepSeek V4 Pro · 14.3s)

_Thinking:_

Very important findings!

1. **vLLM already supports DeepSeek V4!** There's a blog post from Apr 24, 2026: "DeepSeek V4 in vLLM: Efficient Long-context Attention" - they already have support for DeepSeek-V4-Pro and DeepSeek-V4-Flash.

2. There's a `vllm/vllm-openai:deepseekv4-cu129` Docker image.

3. There are several GitHub issues about V4 support with specific hardware:
   - #40778: DeepSeek V4 support request
   - #40928: SM120 support with Triton fallback
   - #40851: sm_80 (A100/A800) support

4. The model files are 160GB for Flash version, 46 safetensors files at ~3.5GB each.

So the current state is:
- vLLM already has DeepSeek V4 support in their main branch
- But this local codebase may be an older version that doesn't have it yet
- The user's specific concern is about the mxfp4 quantization support

Let me check if this local codebase has the deepseek_v4 model file.

Also, looking at the issues, the main challenges are:
1. DeepGEMM/HyperConnection kernels that require specific GPU architectures (sm_90, sm_100, sm_120)
2. The model needs specialized CUDA kernels that may not be available for all GPU architectures

For the user's RTX PRO 2000 (Blackwell architecture, 16GB VRAM), the question is:
1. What SM architecture does it have? RTX PRO 2000 is likely sm_120 or sm_100
2. Even if the kernels work, 16GB VRAM is extremely tight for a 284B model

Let me check if this codebase has deepseek_v4 model file and where the upstream vLLM has it.

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/deepseek_v4*"
}
```

**Output:**
```
No files found
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/deepseek*v4*"
}
```

**Output:**
```
No files found
```

**Tool: grep**

**Input:**
```json
{
  "pattern": "deepseek_v4|DeepseekV4|deepseekv4",
  "include": "*.py",
  "path": "/Users/ross/Documents/project/Lvllm/vllm"
}
```

**Output:**
```
No files found
```

---

## Assistant (Build · DeepSeek V4 Pro · 31.5s)

_Thinking:_

So this local codebase does NOT have DeepSeek V4 support at all. It's an older version of vLLM that predates the V4 model support.

The upstream vLLM project has already added DeepSeek V4 support (as of April 2026). But this local copy doesn't have it.

Now let me think about what the user needs:

1. This is a fork/copy of vLLM that seems to be modified (it has AGENTS.md with custom instructions)
2. They want to add DeepSeek V4 support
3. Specifically they want to handle the mxfp4 quantization
4. Target hardware: 192GB RAM + 16GB VRAM (RTX PRO 2000)

The "NUMA plugin" reference is interesting. Let me check if there's a custom NUMA plugin in this codebase that's different from upstream.

From the exploration:
- NUMA support is inline in `multiproc_executor.py` - it wraps worker processes with `numactl --interleave=all`
- This is controlled by `LVLLM_ENABLE_NUMA_INTERLEAVE` env var (note: LVLLM not VLLM - this might be a custom fork)

Wait! The env var is `LVLLM_ENABLE_NUMA_INTERLEAVE` - this looks like a custom modification. The project is called "Lvllm" not "vLLM". This is likely a fork with custom changes including the NUMA interleaving.

So the user's question is about adding DeepSeek V4 support to this Lvllm fork, which:
1. Doesn't have DeepSeek V4 model support yet
2. Has a custom NUMA interleaving feature
3. May need to handle the mixed FP4+FP8 quantization

The key challenge is: DeepSeek V4 Pro uses MXFP4 (or NVFP4) for MoE expert parameters, but the config.json only declares FP8 quantization. The model loading needs to handle this mixed precision correctly.

Let me now think about the practical implementation plan:

### Challenge 1: Model Architecture Support
The V4 architecture is significantly different from V2/V3:
- Hybrid attention (CSA + HCA)
- mHC connections
- Different attention mechanism (head_dim=512, different from MLA's kv_lora_rank/q_lora_rank approach)
- Sparse indexer with different parameters

This is a major undertaking. The upstream vLLM has already done this work in their `deepseek_v4.py` module and associated CUDA kernels.

### Challenge 2: Mixed FP4+FP8 Quantization
The config.json has `quant_method: "fp8"` with `fmt: "e4m3"` and `scale_fmt: "ue8m0"` which maps to block-scaled FP8 (MXFP8-like). But the actual MoE expert weights are in FP4 format.

Looking at the weight file sizes: 46 files at ~3.5GB each = ~160GB for 284B params. 284B params in bf16 would be ~568GB. In FP8 it would be ~284GB. In mixed FP4+FP8 it could be around 160GB (which matches).

So the weights are already quantized in the safetensors files. The question is what format they're stored in:
- If the config says fp8 with e4m3 fmt and block size 128, the dense weights are likely block-quantized FP8
- The MoE weights must be in a different format (FP4) - this would need a per-layer quantization config

### Challenge 3: Memory Constraints (16GB VRAM)
Even the Flash version (284B total, 13B activated) will need:
- CPU offloading for most parameters
- Only the active layers in GPU memory
- Aggressive KV cache management
- Possible pipeline parallelism

The 192GB system RAM is enough to hold the full model, but 16GB VRAM means only small portions can be on GPU at a time.

Now, the big question is: should I try to implement this from scratch, or should I recommend rebasing on the upstream vLLM that already has V4 support?

Given that:
1. Upstream vLLM already has V4 support
2. This is a fork (Lvllm) with custom NUMA features
3. The user wants to add V4 support to this fork specifically

I think the best approach is:
1. Check if the upstream vLLM V4 support can be cherry-picked/merged
2. If not, analyze what needs to be implemented
3. Focus on the MXFP4 quantization aspect specifically

Let me now look at the upstream vLLM's deepseek_v4.py to understand what they implemented.

**Tool: tavily_tavily_search**

**Input:**
```json
{
  "query": "site:github.com/vllm-project/vllm deepseek_v4.py model file blob main",
  "search_depth": "advanced"
}
```

**Output:**
```
Detailed Results:

Title: [Feature]: deepseek v4 support · Issue #40778 · vllm-project/vllm
URL: https://github.com/vllm-project/vllm/issues/40778
Content: (APIServer pid=1) INFO 04-24 10:27:00 [utils.py:233] non-default args: {'model_tag': '/models/DeepSeek-V4-Flash', 'enable_auto_tool_choice': True, 'tool_call_parser': 'deepseek_v4', 'host': '0.0.0.0', 'model': '/models/DeepSeek-V4-Flash', 'tokenizer_mode': 'deepseek_v4', 'trust_remote_code': True, 'served_model_name': ['VLLM-MODEL'], 'reasoning_parser': 'deepseek_v4', 'tensor_parallel_size': 2, 'enable_expert_parallel': True, 'block_size': 256, 'gpu_memory_utilization': 0.95, 'kv_cache_dtype': 'fp8', 'enable_prefix_caching': True, 'max_num_seqs': 2, 'async_scheduling': True, 'attention_config': AttentionConfig(backend=None, flash_attn_version=None, use_prefill_decode_attention=False, flash_attn_max_num_splits_for_cuda_graph=32, use_cudnn_prefill=False, [...] (Worker_TP0_EP0 pid=408) ERROR 04-24 10:50:25 [multiproc_executor.py:971] File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py", line 833, in forward

 (Worker_TP0_EP0 pid=408) ERROR 04-24 10:50:25 [multiproc_executor.py:971] hidden_states = self.model(

 (Worker_TP0_EP0 pid=408) ERROR 04-24 10:50:25 [multiproc_executor.py:971] ^^^^^^^^^^^

 (Worker_TP0_EP0 pid=408) ERROR 04-24 10:50:25 [multiproc_executor.py:971] File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 611, in call

 (Worker_TP0_EP0 pid=408) ERROR 04-24 10:50:25 [multiproc_executor.py:971] self.aot_compiled_fn = self.aot_compile(args, kwargs)

 (Worker_TP0_EP0 pid=408) ERROR 04-24 10:50:25 [multiproc_executor.py:971] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [...] vllm  | (Worker_TP0_EP0 pid=407) ERROR 04-24 18:37:45 [multiproc_executor.py:971]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
vllm  | (Worker_TP0_EP0 pid=407) ERROR 04-24 18:37:45 [multiproc_executor.py:971]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py", line 833, in forward
vllm  | (Worker_TP0_EP0 pid=407) ERROR 04-24 18:37:45 [multiproc_executor.py:971]     hidden_states = self.model(
vllm  | (Worker_TP0_EP0 pid=407) ERROR 04-24 18:37:45 [multiproc_executor.py:971]                     ^^^^^^^^^^^
vllm  | (Worker_TP0_EP0 pid=407) ERROR 04-24 18:37:45 [multiproc_executor.py:971]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 467, in __call__

Title: [Bug]: Using H200 to deploy DeepSeekV4, after sending a long text, all subsequent requests are blocked. · Issue #40863 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/40863
Content: ### 🐛 Describe the bug

1.deploy the model on H200：  
docker run --gpus all   
--privileged --ipc=host -p 8000:8000   
-v ~/.cache/huggingface:/root/.cache/huggingface   
-e VLLM\_ENGINE\_READY\_TIMEOUT\_S=3600   
vllm/vllm-openai:deepseekv4-cu129 deepseek-ai/DeepSeek-V4-Pro   
--trust-remote-code   
--kv-cache-dtype fp8   
--block-size 256   
--enable-expert-parallel   
--data-parallel-size 8   
--max-model-len 800000   
--gpu-memory-utilization 0.95   
--max-num-seqs 512   
--max-num-batched-tokens 512   
--no-enable-flashinfer-autotune   
--compilation-config '{"mode": 0, "cudagraph\_mode": "FULL\_DECODE\_ONLY"}'   
--tokenizer-mode deepseek\_v4   
--tool-call-parser deepseek\_v4   
--enable-auto-tool-choice   
--reasoning-parser deepseek\_v4 [...] System Info

OS : Ubuntu 22.04.5 LTS (x86_64)
GCC version : (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Clang version : Could not collect
CMake version : Could not collect
Libc version : glibc-2.35

PyTorch Info

PyTorch version : 2.11.0+cu129
Is debug build : False
CUDA used to build PyTorch : 12.9
ROCM used to build PyTorch : N/A
XPU used to build PyTorch : N/A

Python Environment

Python version : 3.12.13 (main, Mar 4 2026, 09:23:07) [GCC 11.4.0] (64-bit runtime)
Python platform : Linux-5.15.0-136-generic-x86_64-with-glibc2.35

CUDA / GPU Info [...] Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves

Title: [Bug]: DeepSeek-V4-Pro H200 DP+EP router dtype mismatch in topk_hash_softplus_sqrt (Long/Int inconsistency) · Issue #40862 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/40862
Content: (Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py", line 833, in forward
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] hidden_states = self.model(
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] ^^^^^^^^^^^
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 467, in __call__
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] return self.forward(args, kwargs)
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [...] (Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] return forward_call(args, kwargs)
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py", line 531, in forward
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] x = self.ffn(x, input_ids)
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] ^^^^^^^^^^^^^^^^^^^^^^ [...] (Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py", line 629, in forward
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] hidden_states = layer(
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] ^^^^^^
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] return self._call_impl(args, kwargs)
(Worker_DP6_EP6 pid=3561) ERROR 04-25 05:03:54 [multiproc_executor.py:971] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Title: 2026-04-24-deepseek-v4.md - GitHub
URL: https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md
Content: These models feature an efficient long-context attention mechanism, purpose-built for tasks involving up to _one million tokens_. While the new attention design may appear intricate on first reading, its underlying principles are straightforward once examined systematically.

This blog post is organized into three sections:

   Quickstart guide for serving DeepSeek V4 on vLLM
   First-principles explanation of DeepSeek V4's new architectural design
   Overview of our implementation approach and optimization challenges for this model on vLLM: hybrid KV cache, kernel fusion, and disaggregated serving. [...] Raw

Copy raw file

Download raw file

You must be signed in to make or propose changes

More edit options

Outline

Edit and raw actions

| layout | post |

| title | DeepSeek V4 in vLLM: Efficient Long-context Attention |
| author | vLLM Team |
| image | /assets/figures/2026-04-dsv4/thumbnail.png |
| summary | A first-principles walkthrough of DeepSeek V4's long-context attention, and how we implemented it in vLLM. |
| tags | model-support |

We are excited to announce that vLLM now supports the DeepSeek V4 family of models (`deepseek-ai/DeepSeek-V4-Pro` and `deepseek-ai/DeepSeek-V4-Flash`). [...] This represents our initial release of model support, and further optimizations are actively underway. We hope the technical explanation that follows can help the open-source community understand both the attention mechanism itself and the rationale behind our current implementation decisions.

## Running DeepSeek V4 on vLLM

, which is substantially more memory-efficient than standard Multi-head Attention (MHA) or Multi-Query Attention (MQA), scaling to one million tokens remains difficult given the limited capacity of GPU memory.
   Attention computation cost: Computing attention over long contexts is expensive. Even with prior techniques such as DeepSeek Sparse Attention (DSA), the computation remains a significant bottleneck.

Title: [Bug]: Title: DeepSeek V4 intermittently leaks DSML fragments in auto + streaming mode, causing unstable tool-call parsing · Issue #40801 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/40801
Content: GPU5 NV18 NV18 NV18 NV18 NV18 X NV18 NV18 SYS SYS SYS SYS SYS NODE 48-95,144-191 1 N/A
GPU6 NV18 NV18 NV18 NV18 NV18 NV18 X NV18 SYS SYS SYS SYS SYS PIX 48-95,144-191 1 N/A
GPU7 NV18 NV18 NV18 NV18 NV18 NV18 NV18 X SYS SYS SYS SYS SYS PIX 48-95,144-191 1 N/A
NIC0 PIX PIX NODE NODE SYS SYS SYS SYS X PIX NODE NODE NODE SYS
NIC1 PIX PIX NODE NODE SYS SYS SYS SYS PIX X NODE NODE NODE SYS
NIC2 NODE NODE NODE NODE SYS SYS SYS SYS NODE NODE X PIX NODE SYS
NIC3 NODE NODE NODE NODE SYS SYS SYS SYS NODE NODE PIX X NODE SYS
NIC4 NODE NODE NODE NODE SYS SYS SYS SYS NODE NODE NODE NODE X SYS
NIC5 SYS SYS SYS SYS NODE NODE PIX PIX SYS SYS SYS SYS SYS X
Legend:
X = Self
SYS = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI) [...] System Info

OS : Ubuntu 22.04.5 LTS (x86_64)
GCC version : (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Clang version : Could not collect
CMake version : Could not collect
Libc version : glibc-2.35

PyTorch Info

PyTorch version : 2.11.0+cu129
Is debug build : False
CUDA used to build PyTorch : 12.9
ROCM used to build PyTorch : N/A
XPU used to build PyTorch : N/A

Python Environment

Python version : 3.12.13 (main, Mar 4 2026, 09:23:07) [GCC 11.4.0] (64-bit runtime)
Python platform : Linux-5.15.0-119-generic-x86_64-with-glibc2.35

CUDA / GPU Info [...] Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1
```

**Tool: webfetch**

**Input:**
```json
{
  "url": "https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md",
  "format": "markdown"
}
```

**Output:**
```
                     vllm-project.github.io/\_posts/2026-04-24-deepseek-v4.md at main · vllm-project/vllm-project.github.io · GitHub                                             

[Skip to content](#start-of-content)   

## Navigation Menu

Toggle navigation

[](/)

[Sign in](/login?return_to=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm-project.github.io%2Fblob%2Fmain%2F_posts%2F2026-04-24-deepseek-v4.md)

Appearance settings

  

-   Platform
    
    -   AI CODE CREATION
        
        -   [
            
            GitHub CopilotWrite better code with AI
            
            ](https://github.com/features/copilot)
        -   [
            
            GitHub SparkBuild and deploy intelligent apps
            
            ](https://github.com/features/spark)
        -   [
            
            GitHub ModelsManage and compare prompts
            
            ](https://github.com/features/models)
        -   [
            
            MCP RegistryNewIntegrate external tools
            
            ](https://github.com/mcp)
        
    -   DEVELOPER WORKFLOWS
        
        -   [
            
            ActionsAutomate any workflow
            
            ](https://github.com/features/actions)
        -   [
            
            CodespacesInstant dev environments
            
            ](https://github.com/features/codespaces)
        -   [
            
            IssuesPlan and track work
            
            ](https://github.com/features/issues)
        -   [
            
            Code ReviewManage code changes
            
            ](https://github.com/features/code-review)
        
    -   APPLICATION SECURITY
        
        -   [
            
            GitHub Advanced SecurityFind and fix vulnerabilities
            
            ](https://github.com/security/advanced-security)
        -   [
            
            Code securitySecure your code as you build
            
            ](https://github.com/security/advanced-security/code-security)
        -   [
            
            Secret protectionStop leaks before they start
            
            ](https://github.com/security/advanced-security/secret-protection)
        
    -   EXPLORE
        
        -   [Why GitHub](https://github.com/why-github)
        -   [Documentation](https://docs.github.com)
        -   [Blog](https://github.blog)
        -   [Changelog](https://github.blog/changelog)
        -   [Marketplace](https://github.com/marketplace)
        
    
    [View all features](https://github.com/features)
    
-   Solutions
    
    -   BY COMPANY SIZE
        
        -   [Enterprises](https://github.com/enterprise)
        -   [Small and medium teams](https://github.com/team)
        -   [Startups](https://github.com/enterprise/startups)
        -   [Nonprofits](https://github.com/solutions/industry/nonprofits)
        
    -   BY USE CASE
        
        -   [App Modernization](https://github.com/solutions/use-case/app-modernization)
        -   [DevSecOps](https://github.com/solutions/use-case/devsecops)
        -   [DevOps](https://github.com/solutions/use-case/devops)
        -   [CI/CD](https://github.com/solutions/use-case/ci-cd)
        -   [View all use cases](https://github.com/solutions/use-case)
        
    -   BY INDUSTRY
        
        -   [Healthcare](https://github.com/solutions/industry/healthcare)
        -   [Financial services](https://github.com/solutions/industry/financial-services)
        -   [Manufacturing](https://github.com/solutions/industry/manufacturing)
        -   [Government](https://github.com/solutions/industry/government)
        -   [View all industries](https://github.com/solutions/industry)
        
    
    [View all solutions](https://github.com/solutions)
    
-   Resources
    
    -   EXPLORE BY TOPIC
        
        -   [AI](https://github.com/resources/articles?topic=ai)
        -   [Software Development](https://github.com/resources/articles?topic=software-development)
        -   [DevOps](https://github.com/resources/articles?topic=devops)
        -   [Security](https://github.com/resources/articles?topic=security)
        -   [View all topics](https://github.com/resources/articles)
        
    -   EXPLORE BY TYPE
        
        -   [Customer stories](https://github.com/customer-stories)
        -   [Events & webinars](https://github.com/resources/events)
        -   [Ebooks & reports](https://github.com/resources/whitepapers)
        -   [Business insights](https://github.com/solutions/executive-insights)
        -   [GitHub Skills](https://skills.github.com)
        
    -   SUPPORT & SERVICES
        
        -   [Documentation](https://docs.github.com)
        -   [Customer support](https://support.github.com)
        -   [Community forum](https://github.com/orgs/community/discussions)
        -   [Trust center](https://github.com/trust-center)
        -   [Partners](https://github.com/partners)
        
    
    [View all resources](https://github.com/resources)
    
-   Open Source
    
    -   COMMUNITY
        
        -   [
            
            GitHub SponsorsFund open source developers
            
            ](https://github.com/sponsors)
        
    -   PROGRAMS
        
        -   [Security Lab](https://securitylab.github.com)
        -   [Maintainer Community](https://maintainers.github.com)
        -   [Accelerator](https://github.com/accelerator)
        -   [GitHub Stars](https://stars.github.com)
        -   [Archive Program](https://archiveprogram.github.com)
        
    -   REPOSITORIES
        
        -   [Topics](https://github.com/topics)
        -   [Trending](https://github.com/trending)
        -   [Collections](https://github.com/collections)
        
    
-   Enterprise
    
    -   ENTERPRISE SOLUTIONS
        
        -   [
            
            Enterprise platformAI-powered developer platform
            
            ](https://github.com/enterprise)
        
    -   AVAILABLE ADD-ONS
        
        -   [
            
            GitHub Advanced SecurityEnterprise-grade security features
            
            ](https://github.com/security/advanced-security)
        -   [
            
            Copilot for BusinessEnterprise-grade AI features
            
            ](https://github.com/features/copilot/copilot-business)
        -   [
            
            Premium SupportEnterprise-grade 24/7 support
            
            ](https://github.com/premium-support)
        
    
-   [Pricing](https://github.com/pricing)

Search or jump to...

# Search code, repositories, users, issues, pull requests...

Search

Clear

[Search syntax tips](https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax)

# Provide feedback

We read every piece of feedback, and take your input very seriously.

 Include my email address so I can be contacted

Cancel Submit feedback

# Saved searches

## Use saved searches to filter your results more quickly

Name  

Query 

To see all available qualifiers, see our [documentation](https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax).

Cancel Create saved search

[Sign in](/login?return_to=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm-project.github.io%2Fblob%2Fmain%2F_posts%2F2026-04-24-deepseek-v4.md)

[Sign up](/signup?ref_cta=Sign+up&ref_loc=header+logged+out&ref_page=%2F%3Cuser-name%3E%2F%3Crepo-name%3E%2Fblob%2Fshow&source=header-repo&source_repo=vllm-project%2Fvllm-project.github.io)

Appearance settings

Resetting focus

You signed in with another tab or window. Reload to refresh your session. You signed out in another tab or window. Reload to refresh your session. You switched accounts on another tab or window. Reload to refresh your session. Dismiss alert

[vllm-project](/vllm-project) / **[vllm-project.github.io](/vllm-project/vllm-project.github.io)** Public

-   [Notifications](/login?return_to=%2Fvllm-project%2Fvllm-project.github.io) You must be signed in to change notification settings
-   [Fork 87](/login?return_to=%2Fvllm-project%2Fvllm-project.github.io)
-   [Star 36](/login?return_to=%2Fvllm-project%2Fvllm-project.github.io)
    

-   [Code](/vllm-project/vllm-project.github.io)
-   [Issues 1](/vllm-project/vllm-project.github.io/issues)
-   [Pull requests 4](/vllm-project/vllm-project.github.io/pulls)
-   [Actions](/vllm-project/vllm-project.github.io/actions)
-   [Security and quality 0](/vllm-project/vllm-project.github.io/security)
-   [Insights](/vllm-project/vllm-project.github.io/pulse)

Additional navigation options

-   [Code](/vllm-project/vllm-project.github.io)
-   [Issues](/vllm-project/vllm-project.github.io/issues)
-   [Pull requests](/vllm-project/vllm-project.github.io/pulls)
-   [Actions](/vllm-project/vllm-project.github.io/actions)
-   [Security and quality](/vllm-project/vllm-project.github.io/security)
-   [Insights](/vllm-project/vllm-project.github.io/pulse)

 [](/vllm-project/vllm-project.github.io)

## FilesExpand file tree

 main

## Breadcrumbs

1.  [vllm-project.github.io](/vllm-project/vllm-project.github.io/tree/main)
2.  /[\_posts](/vllm-project/vllm-project.github.io/tree/main/_posts)

/

# 2026-04-24-deepseek-v4.md

Copy path

Blame

More file actions

Blame

More file actions

## Latest commit

## History

[History](/vllm-project/vllm-project.github.io/commits/main/_posts/2026-04-24-deepseek-v4.md)

[](/vllm-project/vllm-project.github.io/commits/main/_posts/2026-04-24-deepseek-v4.md)History

executable file

·

293 lines (195 loc) · 22.6 KB

 main

## Breadcrumbs

1.  [vllm-project.github.io](/vllm-project/vllm-project.github.io/tree/main)
2.  /[\_posts](/vllm-project/vllm-project.github.io/tree/main/_posts)

/

# 2026-04-24-deepseek-v4.md

Top

## File metadata and controls

-   Preview
    
-   Code
    
-   Blame
    

executable file

·

293 lines (195 loc) · 22.6 KB

[Raw](https://github.com/vllm-project/vllm-project.github.io/raw/refs/heads/main/_posts/2026-04-24-deepseek-v4.md)

Copy raw file

Download raw file

Outline

Edit and raw actions

layout

post

title

DeepSeek V4 in vLLM: Efficient Long-context Attention

author

vLLM Team

image

/assets/figures/2026-04-dsv4/thumbnail.png

summary

A first-principles walkthrough of DeepSeek V4's long-context attention, and how we implemented it in vLLM.

tags

model-support

We are excited to announce that vLLM now supports the DeepSeek V4 family of models ([`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) and [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)).

These models feature an efficient long-context attention mechanism, purpose-built for tasks involving up to *one million tokens*. While the new attention design may appear intricate on first reading, its underlying principles are straightforward once examined systematically.

This blog post is organized into three sections:

-   Quickstart guide for serving DeepSeek V4 on vLLM
-   First-principles explanation of DeepSeek V4's new architectural design
-   Overview of our implementation approach and optimization challenges for this model on vLLM: hybrid KV cache, kernel fusion, and disaggregated serving.

This represents our initial release of model support, and further optimizations are actively underway. We hope the technical explanation that follows can help the open-source community understand both the attention mechanism itself and the rationale behind our current implementation decisions.

## Running DeepSeek V4 on vLLM

[](#running-deepseek-v4-on-vllm)

DeepSeek V4 comes with 2 models, a big 1.6T parameter `DeepSeek-V4-Pro`, and a small 285B parameter `DeepSeek-V4-Flash`. Both models support up to 1 million tokens of context, and vLLM's implementation of the new attention mechanism is designed to scale to that context length.

### DeepSeek-V4-Pro

[](#deepseek-v4-pro)

Here we highlight a single node deployment optimized for easy testing and prototyping, with several optional optimizations like FP4 indexer and MTP. The following command is runnable on 8xB200 or 8xB300.

docker run --gpus all \\
  --ipc=host -p 8000:8000 \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Pro \\
  --trust-remote-code \\
  --kv-cache-dtype fp8 \\
  --block-size 256 \\
  --enable-expert-parallel \\
  --data-parallel-size 8 \\
  --compilation-config '{"cudagraph\_mode":"FULL\_AND\_PIECEWISE", "custom\_ops":\["all"\]}' \\
  --attention\_config.use\_fp4\_indexer\_cache=True \\
  --tokenizer-mode deepseek\_v4 \\
  --tool-call-parser deepseek\_v4 \\
  --enable-auto-tool-choice \\
  --reasoning-parser deepseek\_v4

For more deployment strategies, including disaggregated serving/more GPU architectures, please refer to the [recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro).

### DeepSeek-V4-Flash

[](#deepseek-v4-flash)

Here we highlight a single node deployment optimized for easy testing and prototyping, with several optional optimizations like FP4 indexer and MTP. The following command is runnable on 4xB200 or 4xB300.

docker run --gpus all \\
  --ipc=host -p 8000:8000 \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Flash \\
  --trust-remote-code \\
  --kv-cache-dtype fp8 \\
  --block-size 256 \\
  --enable-expert-parallel \\
  --data-parallel-size 4 \\
  --compilation-config '{"cudagraph\_mode":"FULL\_AND\_PIECEWISE", "custom\_ops":\["all"\]}' \\
  --attention\_config.use\_fp4\_indexer\_cache=True \\
  --tokenizer-mode deepseek\_v4 \\
  --tool-call-parser deepseek\_v4 \\
  --enable-auto-tool-choice \\
  --reasoning-parser deepseek\_v4

For more deployment strategies, including disaggregated serving/more GPU architectures, please refer to the [recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash).

## DeepSeek V4's Attention Mechanism Explained

[](#deepseek-v4s-attention-mechanism-explained)

Long-context inference faces two main challenges:

-   **KV cache memory growth**: The KV cache scales linearly with context length. While DeepSeek-style models use [Multi-head Latent Attention (MLA)](https://arxiv.org/abs/2405.04434), which is substantially more memory-efficient than standard Multi-head Attention (MHA) or Multi-Query Attention (MQA), scaling to one million tokens remains difficult given the limited capacity of GPU memory.
-   **Attention computation cost**: Computing attention over long contexts is expensive. Even with prior techniques such as [DeepSeek Sparse Attention (DSA)](http://arxiv.org/abs/2512.02556), the computation remains a significant bottleneck.

To address these challenges, the DeepSeek team designed a new attention mechanism aimed at both compressing the KV cache and reducing attention computation time.

1.  Share key and value vectors (2x memory savings). For correctness, we apply an **inverse RoPE** operation to the attention output.
2.  Compress the KV cache across multiple tokens (4x to 128x memory savings). In DeepSeek V4, there are two ways to do this:
    -   **`c4a`**: compress the KV cache by roughly 1/4. One compressed token is a weighted sum of **8 uncompressed tokens**, with a **stride of 4**.
    -   **`c128a`**: compress the KV cache by roughly 1/128. One compressed token is a weighted sum of **128 uncompressed tokens**, with a **stride of 128**.
3.  DeepSeek Sparse Attention (bounded attention computation cost). Even after compressing the KV cache with `c4a` attention, a one-million-token sequence still has 250k compressed tokens. To accelerate the attention computation, we can use [DeepSeek Sparse Attention (DSA)](http://arxiv.org/abs/2512.02556) to attend to only top-$k$ compressed tokens.
4.  Preserving locality: Short sliding window. DeepSeek V4 uses a sliding window of size 128 for local information, operating on the uncompressed tokens, so that a query token can attend to local information before it reaches the compression boundary.

To better illustrate this new attention mechanism, here's an animation of the `c4a` attention processing 13 tokens. With the details above in mind, the `c128a` case should be straightforward to follow as well. Launch the [interactive version](/vllm-project/vllm-project.github.io/blob/main/assets/interactive_pages/c4a.html) to hover over tokens and inspect the connections.

[![](/vllm-project/vllm-project.github.io/raw/main/assets/figures/2026-04-dsv4/c4a_animation.gif)](/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-04-dsv4/c4a_animation.gif)  
*Animation of c4a attention*

The efficient attention design leads to substantial KV cache savings. With `bf16` KV cache, DeepSeek V4 only has 9.62 GiB KV cache per sequence at 1M context. That is about 8.7x smaller than the 83.9 GiB estimate for a 61-layer DeepSeek V3.2-style stack. In practice, we use `fp4` for the indexer cache and `fp8` for the attention cache, which further reduces the KV cache size by roughly **2x** compared to the `bf16` estimate!

[![](/vllm-project/vllm-project.github.io/raw/main/assets/figures/2026-04-dsv4/kv-cache-comparison.svg)](/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-04-dsv4/kv-cache-comparison.svg)  
*Per-layer KV state in DeepSeek V3.2 versus DeepSeek V4.*

For more detail on the arithmetic and the mathematical interpretation, please refer to the appendix.

## vLLM's Implementation of DeepSeek V4

[](#vllms-implementation-of-deepseek-v4)

Despite the structural savings, the attention mechanism still carries intrinsic complexity, and realizing those savings efficiently in vLLM is a systems problem with several implementation challenges:

-   Similar to the DeepSeek V3.2 model, the attention kernel uses bfloat16 KV cache for prefill and partially token-wise fp8 for decode.
-   The model uses a mix of `c4a` and `c128a` attention, and some attention layers use purely a sliding window for local information without compression. The heterogeneous attention types make KV cache management much more complex.
-   When batching multiple sequences, they might have different states with respect to the KV cache compression boundary.
-   The model ships with native fp4 MoE weights, which require special handling in vLLM.

Aside from the attention mechanism itself, there are several other updates, including architecture changes like [Manifold-Constrained Hyper-Connections](http://arxiv.org/abs/2512.24880), and some changes to the MoE module. They are not covered in this post, as they are simpler model changes that are easier to adapt.

vLLM addresses these challenges with optimizations on two fronts: memory management and kernel efficiency.

### Keeping the KV Cache Memory Packed

[](#keeping-the-kv-cache-memory-packed)

vLLM's KV cache memory allocator has to pack several kinds of KV state tightly in GPU memory while still working with prefix caching, prefill/decode disaggregation, CUDA graphs, and the rest of vLLM's serving path. Three design choices keep this manageable.

#### (1) A single logical block size

[](#1-a-single-logical-block-size)

Different layers compress at different rates (1/4 for `c4a`, 1/128 for `c128a`, 1/1 for SWA). An obvious design is to size each layer's block around a round number of *compressed* entries. But then every layer gets its own page layout, and the allocator has to reason about all of them separately.

Instead, we fix the logical block at **256 native token positions** for every compressed layer. A `c4a` block then physically holds `256 / 4 = 64` compressed entries, and a `c128a` block holds `256 / 128 = 2`. Allocating a block always means reserving the next 256 native positions of a request's context, regardless of which layer owns it. Slot mapping, scheduler accounting, and prefix-hit detection can all use that same unit instead of branching on `compress_ratio`.

#### (2) Compressor state as a sliding window

[](#2-compressor-state-as-a-sliding-window)

Each compressor layer also maintains a small rolling residual per request: an 8-token (overlapped) partial state for C4, and a 128-token partial state for C128. A natural first design is to keep that residual in a per-request side buffer. That works in isolation, but it becomes awkward once it has to interact with the rest of the serving stack.

With a side buffer, prefix caching would need to snapshot the rolling state at every cacheable boundary, key it alongside the prefix hash, and restore it on a hit. Disaggregated prefill would need a second transfer path that ships residuals from prefill workers to decode workers alongside the KV blocks. Each requirement is manageable on its own, but together they create another state-management path to maintain across features.

vLLM avoids this by treating the compressor state like sliding-window KV. The runtime invariant is the same: fixed size per request, advanced as decoding proceeds, with state outside the window either discarded or handled through caching. So we register the compressor state under the sliding-window KV cache spec, with `sliding_window = coff * compress_ratio` (8 for C4 and 128 for C128), and place it into SWA-style blocks under the same hybrid KV cache manager.

This lets several serving features reuse the same abstraction:

-   **Prefix caching** reuses the normal block semantics. A cache hit lands on a KV cache block boundary (the 256-position unit above), and the compressor state at that boundary is already the correct handoff point.
-   **Disaggregated prefill** treats the compressor state like SWA state. Only the blocks inside the window are transferred, which preserves the transfer-size savings without introducing a separate residual-specific transfer path.
-   **CUDA graphs** and **MTP** follow the same integration pattern as SWA, while keeping metadata and implementation details specific to the compressor state.

#### (3) Unifying page sizes

[](#3-unifying-page-sizes)

The two choices above are still not enough. A C4 indexer block, a `c128a` KV block, and a `c4a` compressor-state block still come in different *page sizes* (different numbers of bytes per block). If each cache kind gets its own block pool, we end up with the same cross-pool fragmentation we were trying to eliminate.

Fortunately, the page size of each cache kind is the product `block_size * compress_ratio * per_entry_size`, and all three factors are under our control. If we choose them carefully, the different cache kinds collapse into a small number of *page-size buckets*, and each bucket can be backed by a single shared block pool.

In our implementation, the entire five-way cache stack fits into **three** page sizes. Each pool is sized once at load time, and allocation becomes a bucket lookup. There is no runtime repartitioning, no per-kind accounting, and no fragmentation between cache kinds.

-   *Largest bucket:* `c4a` main KV, SWA KV, `c4a` compressor state, `c128a` compressor state.
-   *Middle bucket:* C4 indexer KV, C4 indexer compressor state.
-   *Smallest bucket:* `c128a` main KV.

### Keeping the GPU Busy

[](#keeping-the-gpu-busy)

Memory layout is only half of the runtime story; the other half is keeping the GPU compute saturated.

vLLM integrates FlashMLA and FlashInfer, which provide optimized attention and MoE kernels. But this model requires many small, mostly memory-bound kernels. We need to avoid extra launches and HBM round-trips that would otherwise slow the full decode path.

[![](/vllm-project/vllm-project.github.io/raw/main/assets/figures/2026-04-dsv4/decode-path.svg)](/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-04-dsv4/decode-path.svg)  
*\`c4a\` decode path: operator graph with kernel fusions (colored outlines) and multi-stream partitioning (default stream = blue band, indexer stream = amber band).*

#### (1) Kernel Fusion

[](#1-kernel-fusion)

We deploy three fusions to cut memory round-trips. In the figure below, these appear as the colored outlines around groups of operators.

-   **Compressor + RMSNorm + RoPE + cache insertion.** After compression, the compressed K immediately goes through RMSNorm, RoPE, and insertion into the following attention's KV cache, either for main attention or for the indexer. Because these stages are almost entirely elementwise, we fuse them into one kernel. We keep separate kernels for the indexer K cache and the main-attention K cache so the parallelization strategy can still be tuned to each head dim. Overall we see a ~1.4-3x speedup over the unfused baseline.
-   **Inverse RoPE + fp8 quant.** After main attention, the output goes through inverse RoPE and then into the fp8 batched matmul for the `o_lora` projection. Fusing the two avoids a back-to-back HBM round trip and raises arithmetic intensity, for a ~2-3x speedup over the unfused version.
-   **Fused Q norm + KV RoPE + K insert.** Before main attention, we need KV cache insertion for both the compressed path and the sliding-window path. The compressed path is already covered by the first fusion, so what remains is elementwise work on the queries and the uncompressed SWA keys. We horizontally fuse that work into a single kernel with static `warpID` dispatch: each warp works independently on either a Q head or a K head, so no cross-warp communication is needed. This delivers a 10-20x speedup over the naive unfused kernels.

We also reuse fusions from our DeepSeek V3.2 work, including Q RoPE + quant + weight multiply, and the horizontal fusion of QK norm right after QK projection at the start of attention.

#### (2) Multi-stream

[](#2-multi-stream)

The operations before main attention are highly parallelizable. They break into three pieces: indexer computation, main-attention KV compression, and sliding-window token insertion. After the initial projection these branches are almost independent, so we overlap them across CUDA streams. The same figure can be read a second way here: the blue band marks the default stream, while the amber band marks the indexer stream.

-   For `c128a` layers, which have no indexer, we run main KV compression in parallel with SWA token insertion.
-   For `c4a` layers, we run the full indexer pipeline on its own stream in parallel with main KV compression and SWA token insertion (the latter two remain serial with respect to each other).

With these overlaps, we observe a 5-6% end-to-end latency reduction at low batch sizes, a useful sign that the decode path spends less time underutilizing the GPU.

On top of that, we use CUDA graphs to cut launch overhead on the decode path, as we do for every other model.

For the full implementation, see the [PR](https://github.com/vllm-project/vllm/pull/40760).

## Planned Work

[](#planned-work)

We are actively working on the following optimizations to further improve the performance of DeepSeek V4 on vLLM:

-   DeepGEMM MegaMoE kernel
-   Paged prefill kernel

The current implementation mainly targets NVIDIA GPUs, including both the Hopper and Blackwell architectures. The deployment recipes for these accelerators can be found at [our recipe website](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro). With vLLM's extensible plugin system, hardware vendors can add support for models directly. For example, [vllm-ascend](https://github.com/vllm-project/vllm-ascend) and [vllm-mlu](https://github.com/Cambricon/vllm-mlu) both support DeepSeek V4 independently.

## Acknowledgments

[](#acknowledgments)

We want to thank the DeepSeek team for open-sourcing DeepSeek V4, as well as DeepSeek leadership for their trust and support in vLLM! The model support is made possible by the contributions from [Inferact Inc.](https://inferact.ai/), a company aiming to grow vLLM as the world's AI inference engine and accelerate AI progress by making inference cheaper and faster.

## Appendix: The Math behind DeepSeek V4's Attention Mechanism

[](#appendix-the-math-behind-deepseek-v4s-attention-mechanism)

### Why inverse RoPE is needed when key and value are shared

[](#why-inverse-rope-is-needed-when-key-and-value-are-shared)

Given a query token at position $i$, the query representation after applying [RoPE](http://arxiv.org/abs/2104.09864) is $&lt;q\_i, i&gt; = R(i)q\_i$, where $R(i)$ is the rotation matrix with the rotation angles parameterized by the position $i$. Some basic properties of the rotation matrix are:

-   $R(i)R(j) = R(i+j)$
-   $R(i)^{-1} = R(i)^T = R(-i)$
-   $R(i)$ is an orthogonal matrix, i.e., $R(i)R(i)^T = I$

Given a set of key tokens at positions $j\_1, j\_2, j\_p, ..., j\_n$, the key representations after applying RoPE are $&lt;k\_{j\_1}, j\_1&gt; = R(j\_1)k\_{j\_1}$, $&lt;k\_{j\_2}, j\_2&gt; = R(j\_2)k\_{j\_2}$, ..., $&lt;k\_{j\_p}, j\_p&gt; = R(j\_p)k\_{j\_p}$, ..., $&lt;k\_{j\_n}, j\_n&gt; = R(j\_n)k\_{j\_n}$.

For value vectors at positions $j\_1, j\_2, j\_p, ..., j\_n$, usually we don't apply RoPE to them. The value representations are simply $&lt;v\_{j\_1}, j\_1&gt; = v\_{j\_1}$, $&lt;v\_{j\_2}, j\_2&gt; = v\_{j\_2}$, ..., $&lt;v\_{j\_p}, j\_p&gt; = v\_{j\_p}$, ..., $&lt;v\_{j\_n}, j\_n&gt; = v\_{j\_n}$.

The attention output is then (omitting some details, such as the scaling factor, for simplicity):

$$ a\_i = \\sum\_{p=1}^n \\frac{\\exp(<q\_i, i>^T <k\_{j\_p}, j\_p>)}{\\sum\_{r=1}^n \\exp(<q\_i, i>^T <k\_{j\_r}, j\_r>)} <v\_{j\_p}, j\_p> = \\sum\_{p=1}^n \\frac{\\exp(q\_i^T R(j\_p - i)k\_{j\_p})}{\\sum\_{r=1}^n \\exp(q\_i^T R(j\_r - i)k\_{j\_r})} v\_{j\_p} $$

One nice property of the attention output is that it is translation invariant. Any factor that depends on position, namely $R(j\_p -i)$ and $R(j\_r -i)$, depends only on the relative position between the query and the key. This means the attention output is the same if we shift the query and the key by the same amount.

If we share the key and value vectors, the attention output will be:

$$ a\_i = \\sum\_{p=1}^n \\frac{\\exp(<q\_i, i>^T <k\_{j\_p}, j\_p>)}{\\sum\_{r=1}^n \\exp(<q\_i, i>^T <k\_{j\_r}, j\_r>)} <k\_{j\_p}, j\_p> = \\sum\_{p=1}^n \\frac{\\exp(q\_i^T R(j\_p -i)k\_{j\_p})}{\\sum\_{r=1}^n \\exp(q\_i^T R(j\_r -i)k\_{j\_r})} R(j\_p) k\_{j\_p} $$

Now the output carries absolute position information through the rotation matrix $R(j\_p)$ directly. This is not what we want. The way to fix it is simple: we apply an inverse RoPE operation to the attention output:

$$ R(-i) a\_i = R(-i) \\sum\_{p=1}^n \\frac{\\exp(<q\_i, i>^T <k\_{j\_p}, j\_p>)}{\\sum\_{r=1}^n \\exp(<q\_i, i>^T <k\_{j\_r}, j\_r>)} <k\_{j\_p}, j\_p> = \\sum\_{p=1}^n \\frac{\\exp(q\_i^T R(j\_p -i)k\_{j\_p})}{\\sum\_{r=1}^n \\exp(q\_i^T R(j\_r -i)k\_{j\_r})} R(j\_p -i) k\_{j\_p} $$

This way, the output only carries relative position information through the rotation matrix $R(j\_p -i)$, and it is translation invariant again.

Similar discussions can be found in [https://kexue.fm/archives/10862](https://kexue.fm/archives/10862) as well.

### Implementation details: exact position ranges and causality conditions

[](#implementation-details-exact-position-ranges-and-causality-conditions)

Care must be taken when processing the compressed KV cache. For each compressed index $j$, we first combine a fixed local group of original tokens, then apply RoPE once using the compressed token's anchor position, and then store that compressed token in the KV cache.

For `c4a`, the $j$\-th compressed token is a weighted sum of tokens in position range $\[4j - 4, 4j + 3\]$, where $j$ starts from 0 and negative indices are treated as tokens with value 0. The position of the compressed token, when we apply RoPE to it, is $4j$.

For `c128a`, the $j$\-th compressed token is a weighted sum of tokens in position range $\[128j, 128j + 127\]$, where $j$ starts from 0. The position of the compressed token, when we apply RoPE to it, is $128j$.

For causality, we need to ensure that a query token at position $i$ only attends to the information produced by tokens in position range $\[0, i\]$. This means that for a query at position $i$ and the $j$\-th compressed token in the KV cache, we need to ensure that $ i \\ge 4j + 3 $ (for `c4a`) or $ i \\ge 128j + 127 $ (for `c128a`).

### Implementation details: The exact value of k in c4a and c128a

[](#implementation-details-the-exact-value-of-k-in-c4a-and-c128a)

For `c4a` attention in DeepSeek V4, the default value of $k$ is 512, and for `c128a` attention, the default value of $k$ is 8192. (For comparison, in DeepSeek V3.2, the default value of $k$ is 2048).

The `c128a` attention has a larger compression ratio. With a 1 million-token context, it possesses at most 8k compressed tokens. 8k tokens are not a big deal for attention computation, so we can simply use full attention over the `c128a` compressed tokens. Implementation-wise, we can still frame the `c128a` attention as a sparse-attention problem whose top-$k$ value is 8192.

### Implementation details: why the short sliding window is needed

[](#implementation-details-why-the-short-sliding-window-is-needed)

With `c128a`, a query token at position $100$ cannot attend to any compressed token in the KV cache, since the first compressed token contains information from position $0$ to $127$, but the query token cannot attend to information after position $100$ due to causality. With the short sliding window, the query token can attend to uncompressed tokens in position range $\[0, 100\]$, so it can still access local information.

### Arithmetic behind the estimates for the 8.7x savings

[](#arithmetic-behind-the-estimates-for-the-87x-savings)

For a sequence with 1M context:

DeepSeek V3.2 with bf16 KV cache:

-   MLA cache per token per layer: $(512 + 64) \\times 2 = 1152$ bytes.
-   Indexer cache per token per layer: $128 \\times 2 = 256$ bytes.
-   Total cached state per token per layer: $1152 + 256 = 1408$ bytes.
-   At 1,048,576 tokens: $1{,}048{,}576 \\times 1408 \\approx 1.375$ GiB per layer.
-   Over 61 layers: about $83.9$ GiB.

DeepSeek V4 at 61 layers with bf16 KV cache:

-   Each shared-KV cached entry stores $512 \\times 2 = 1024$ bytes.
-   Each `c4a` indexer cached entry stores $128 \\times 2 = 256$ bytes.
-   `c4a` layer: shared-KV cache $(128 + 1{,}048{,}576 / 4) \\times 1024$ bytes plus indexer cache $(1{,}048{,}576 / 4) \\times 256$ bytes, for a total of about $320.1$ MiB.
-   `c128a` layer: $(128 + 1{,}048{,}576 / 128) \\times 1024 \\approx 8.1$ MiB.
-   Total across 30 `c4a` layers and 31 `c128a` layers: about $9.62$ GiB.

## Footer

[](https://github.com)© 2026 GitHub, Inc.

### Footer navigation

-   [Terms](https://docs.github.com/site-policy/github-terms/github-terms-of-service)
-   [Privacy](https://docs.github.com/site-policy/privacy-policies/github-privacy-statement)
-   [Security](https://github.com/security)
-   [Status](https://www.githubstatus.com/)
-   [Community](https://github.community/)
-   [Docs](https://docs.github.com/)
-   [Contact](https://support.github.com?tags=dotcom-footer)
-   Manage cookies
-   Do not share my personal information

You can’t perform that action at this time.
```

---

## Assistant (Build · DeepSeek V4 Pro · 418.2s)

_Thinking:_

Now I have a very complete picture. Let me summarize what I know:

1. **Upstream vLLM already supports DeepSeek V4** as of April 24, 2026 via PR #40760. The model is in `vllm/model_executor/models/deepseek_v4.py`.

2. **This local Lvllm fork** does NOT have the deepseek_v4.py file - it only has deepseek_v2.py.

3. **The V4 model** uses:
   - Hybrid attention: c4a (compress 4x) + c128a (compress 128x) + sliding window
   - mHC (Manifold-Constrained Hyper-Connections)
   - FP4 MoE weights natively
   - FP4 indexer cache option (controlled by `attention_config.use_fp4_indexer_cache`)
   - Shared KV across attention layers
   - New attention mechanism with inverse RoPE
   - tokenizer_mode: "deepseek_v4"

4. **NVFP4 vs MXFP4**: The V4 model "ships with native fp4 MoE weights" - these could be either NVFP4 or MXFP4 format. Given that the config.json has `quantization_config.quant_method: "fp8"`, the FP4 aspect is likely at the per-weight level, not declared in the config.json quantization config.

5. **The user's target hardware**: RTX PRO 2000 with 16GB VRAM + 192GB system RAM. This is very constrained for DeepSeek V4 models.

Now, here's my assessment for the user:

The "NUMA plugin" they refer to is actually the inline NUMA interleaving support in `multiproc_executor.py` (which uses `LVLLM_ENABLE_NUMA_INTERLEAVE` env var). But the real issue is:
1. DeepSeek V4 model is not in this codebase
2. Even if it was, 16GB VRAM is extremely tight
3. The MXFP4 support exists in vLLM but may need adaptation for V4's specific weight format

For the actual implementation, the user needs to understand:
- This is a major undertaking (hundreds/thousands of lines of model code + custom CUDA kernels)
- Upstream vLLM already did this work
- The better approach is to rebase on upstream or port the upstream V4 support

Let me now write up a comprehensive analysis and actionable plan. I should focus on:
1. What needs to happen technically
2. The mxfp4 quantization challenge specifically  
3. The memory constraints and how to handle them
4. Practical next steps

Let me look at one more thing - the actual weight format. The blog post says "the model ships with native fp4 MoE weights." Let me check if there's any special handling for this in the upstream code.

Actually, looking at the error messages from the GitHub issues:
```
File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py", line 833
File ".../vllm/model_executor/models/deepseek_v4.py", line 531
File ".../vllm/model_executor/models/deepseek_v4.py", line 629
```

This confirms upstream has a `deepseek_v4.py` file. The model is live.

Now let me think about what the user specifically needs to do given their hardware constraints.

DeepSeek-V4-Flash (284B total, 13B activated):
- 160GB total download size (46 safetensors files)
- To fit in 16GB VRAM: need expert offloading, layer-by-layer loading
- The NUMA interleaving can help with CPU memory bandwidth when offloading

The key issue with mxfp4:
- V4 MoE weights are stored as FP4 in the safetensors
- vLLM's `Mxfp4Config` supports MXFP4 MoE but needs explicit `--quantization mxfp4`
- But the config.json says `quant_method: "fp8"`, so auto-detection would pick FP8
- The model probably needs a custom quantization config that handles the mixed FP4+FP8
- Or the weight loaders need to handle the FP4 format natively

Let me write my final analysis now.

Now I have a complete picture. Here's my analysis:

---

## Analysis: DeepSeek V4 Pro Support for Lvllm

### Current State

- **This Lvllm codebase** does NOT have DeepSeek V4 support — no `deepseek_v4.py` exists
- **Upstream vLLM** added V4 support on April 24, 2026 (PR #40760) via `vllm/model_executor/models/deepseek_v4.py`
- The model is served via Docker image `vllm/vllm-openai:deepseekv4-cu129`

### V4 Architecture Changes (vs V2/V3)

| Feature | V2/V3 | V4 |
|---|---|---|
| `model_type` | `deepseek_v2` | `deepseek_v4` |
| `architectures` | `DeepseekV2ForCausalLM` | `DeepseekV4ForCausalLM` |
| Attention | MLA + DSA | c4a + c128a hybrid compression + sliding window |
| KV sharing | No | Shared K/V with inverse RoPE |
| Residual connections | Standard | mHC (Manifold-Constrained Hyper-Connections) |
| MoE quant | FP8/bf16 | **Native FP4 MoE weights** + FP8 dense |
| Config params | q_lora_rank, kv_lora_rank, etc. | `compress_ratios`, `hc_eps`, `hc_mult`, `hc_sinkhorn_iters`, `scoring_func`, `swiglu_limit` |

### The MXFP4 Challenge

The config.json declares `quantization_config.quant_method: "fp8"` (block-scaled FP8, e4m3 + ue8m0, block_size 128). But MoE expert weights are natively stored in FP4. This means:

1. **Auto-detection mismatch**: vLLM's `_verify_quantization()` will detect `quant_method: "fp8"` and select `ModelOptFp8Config`. But when it hits MoE layers, the weights are FP4 — loading fails.

2. **Existing FP4 paths don't cover V4**: The `Mxfp4Config` has `get_config_filenames()` returning `[]` — no auto-detection. And the V4's FP4 format may differ from either `mxfp4.py` (OCP MX format, block_size 32) or `modelopt.py` (NVFP4, block_size 16).

3. **The upstream vLLM V4 implementation** likely handles this with a custom quant config or weight loading path specific to DeepSeek V4.

### Memory Analysis: 192GB RAM + 16GB VRAM (RTX PRO 2000)

| Component | Size |
|---|---|
| DeepSeek-V4-Flash (mixed FP4+FP8) | ~160GB disk |
| DeepSeek-V4-Pro (mixed FP4+FP8) | ~685GB disk |
| V4-Flash activated params (13B) | ~13-26GB in GPU |
| KV cache per 1M-token seq (fp8) | ~4.8GB |

**16GB VRAM can barely fit V4-Flash with:**
- CPU offloading via `--cpu-offload-gb` or pipeline parallelism
- FP8 KV cache (`--kv-cache-dtype fp8`)
- Small context window (not 1M tokens)
- Expert offloading (the 256 experts don't all fit in VRAM)

The NUMA interleaving (`LVLLM_ENABLE_NUMA_INTERLEAVE`) helps here by spreading CPU memory allocations across NUMA nodes, improving bandwidth when weights are offloaded to system RAM.

### What Needs to Be Done

There are **three layers** of work:

**Layer 1: Model Architecture** (`deepseek_v4.py`) — The hard part
- New attention: c4a/c128a compression kernels, inverse RoPE, shared KV
- mHC connections replacing standard residuals
- New indexer/sparse attention
- Registration in `registry.py` for `"deepseek_v4"` → `DeepseekV4ForCausalLM`

**Layer 2: Mixed FP4+FP8 Quantization** — The specific mxfp4 problem
- Either extend `ModelOptMixedPrecisionConfig` to support per-layer MXFP4
- Or create a `DeepseekV4QuantConfig` that handles V4's specific weight format
- The weight loader needs to detect FP4 tensors (likely packed uint8 with block scales) and route them correctly

**Layer 3: Memory Optimization for 16GB VRAM** — Runtime
- Pipeline parallelism (PP) with CPU offloading
- Expert parallelism (EP) where each GPU handles a subset of experts
- The NUMA interleaving plugin helps CPU-side memory bandwidth

### Recommended Approach

Since upstream vLLM has already implemented V4 support, the most practical path is:

1. **Cherry-pick/port** the upstream `deepseek_v4.py` and related files into this Lvllm fork
2. **Adapt** the quantization handling to work with the existing `Mxfp4Config`/`ModelOptNvFp4Config` infrastructure
3. **Configure** the NUMA interleaving for CPU offloading
4. **Test** with DeepSeek-V4-Flash first (smaller model)

**Tool: question**

**Input:**
```json
{
  "questions": [
    {
      "question": "I found that upstream vLLM already has DeepSeek V4 support (merged April 24). Your local Lvllm fork predates this. Do you want me to:\n1. Port the upstream deepseek_v4.py and related files into your fork?\n2. Focus specifically on the MXFP4 quantization loading path?\n3. First look at the upstream implementation to understand the FP4 weight format before deciding?",
      "header": "Approach",
      "options": [
        {
          "label": "Port upstream V4 code",
          "description": "Cherry-pick/port the upstream deepseek_v4.py model implementation into this Lvllm fork"
        },
        {
          "label": "Focus on MXFP4 loading",
          "description": "Concentrate on the quantization weight loading path - understanding how V4's FP4 MoE weights are stored and making them loadable"
        },
        {
          "label": "Study upstream first",
          "description": "Fetch the upstream V4 implementation to understand the FP4 format and architecture before deciding on approach"
        }
      ]
    },
    {
      "question": "Which model are you targeting for your 16GB VRAM system?",
      "header": "Target model",
      "options": [
        {
          "label": "DeepSeek-V4-Flash",
          "description": "284B total, 13B activated — more feasible for 16GB VRAM"
        },
        {
          "label": "DeepSeek-V4-Pro",
          "description": "1.6T total, 49B activated — extremely challenging for 16GB VRAM"
        }
      ]
    }
  ]
}
```

**Output:**
```
User has answered your questions: "I found that upstream vLLM already has DeepSeek V4 support (merged April 24). Your local Lvllm fork predates this. Do you want me to:
1. Port the upstream deepseek_v4.py and related files into your fork?
2. Focus specifically on the MXFP4 quantization loading path?
3. First look at the upstream implementation to understand the FP4 weight format before deciding?"="Study upstream first", "Which model are you targeting for your 16GB VRAM system?"="DeepSeek-V4-Flash". You can now continue with the user's answers in mind.
```

---

## Assistant (Build · DeepSeek V4 Pro · 12.6s)

_Thinking:_

The user wants to:
1. Study the upstream V4 implementation first to understand the FP4 format
2. Target DeepSeek-V4-Flash

Let me fetch the upstream deepseek_v4.py and related files to understand how they handle the FP4 weights.

Let me study the upstream V4 implementation to understand the FP4 weight format and architecture.

**Tool: webfetch**

**Input:**
```json
{
  "format": "markdown",
  "url": "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/deepseek_v4.py",
  "max_length": 80000
}
```

**Error:**
```
StatusCode: non 2xx status code (404 GET https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/deepseek_v4.py)
```

**Tool: tavily_tavily_search**

**Input:**
```json
{
  "query": "site:github.com/vllm-project/vllm deepseek_v4.py \"fp4\" OR \"mxfp4\" OR \"nvfp4\" quantization weight loading",
  "search_depth": "advanced"
}
```

**Output:**
```
Detailed Results:

Title: [Feature]: Deepseek V4 cannot run ,Please support SM120 GPU,example rtx5090  rtxpro6000 · Issue #40802 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/40802
Content: Loading safetensors checkpoint shards: 100% Completed | 46/46 [21:30<00:00, 46.37s/it]  
Loading safetensors checkpoint shards: 100% Completed | 46/46 [21:30<00:00, 28.07s/it]  
(Worker\_TP0\_EP0 pid=408)  
(Worker\_TP0\_EP0 pid=408) INFO 04-24 10:49:25 [default\_loader.py:384] Loading weights took 1292.19 seconds  
(Worker\_TP0\_EP0 pid=408) INFO 04-24 10:49:27 [mxfp4.py:1238] Using MoEPrepareAndFinalizeNoDPEPModular  
(Worker\_TP0\_EP0 pid=408) INFO 04-24 10:50:00 [gpu\_model\_runner.py:4848] Model loading took 74.05 GiB memory and 1307.649939 seconds  
(Worker\_TP0\_EP0 pid=408) INFO 04-24 10:50:20 [backends.py:1070] Using cache directory: /root/.cache/vllm/torch\_compile\_cache/6df291e80d/rank\_0\_0/backbone for vLLM's torch.compile [...] (Worker\_TP0\_EP0 pid=408) INFO 04-24 10:27:50 [mxfp4.py:481] Using 'MARLIN' Mxfp4 MoE backend.  
(Worker\_TP0\_EP0 pid=408) INFO 04-24 10:27:50 [deepseek\_v4\_attention.py:969] Using MXFP4 indexer cache for Lighening Indexer.  
(Worker\_TP0\_EP0 pid=408) INFO 04-24 10:27:54 [weight\_utils.py:904] Filesystem type for checkpoints: 9P. Checkpoint size: 148.66 GiB. Available RAM: 0.65 GiB.  
(Worker\_TP0\_EP0 pid=408) INFO 04-24 10:27:54 [weight\_utils.py:934] Auto-prefetch is disabled because the filesystem (9P) is not a recognized network FS (NFS/Lustre) and the checkpoint size (148.66 GiB) exceeds 90% of available RAM (0.65 GiB).  
Loading safetensors checkpoint shards: 4% Completed | 2/46 [00:33<13:38, 18.60s/it] [...] (APIServer pid=1) INFO 04-24 10:27:00 [config.py:763] Detected quantization\_config.scale\_fmt=ue8m0; enabling UE8M0 for DeepGEMM.  
(APIServer pid=1) INFO 04-24 10:27:00 [config.py:449] Replacing legacy 'type' key with 'rope\_type'  
(APIServer pid=1) INFO 04-24 10:27:11 [model.py:555] Resolved architecture: DeepseekV4ForCausalLM  
(APIServer pid=1) INFO 04-24 10:27:11 [model.py:1689] Using max model len 1048576  
(APIServer pid=1) INFO 04-24 10:27:11 [cache.py:267] Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor  
(APIServer pid=1) INFO 04-24 10:27:11 [scheduler.py:238] Chunked prefill is enabled with max\_num\_batched\_tokens=8192.

Title: [Feature]: deepseek v4 support · Issue #40778 · vllm-project/vllm
URL: https://github.com/vllm-project/vllm/issues/40778
Content: Loading safetensors checkpoint shards: 100% Completed | 46/46 [21:30<00:00, 28.07s/it]

 (Worker_TP0_EP0 pid=408)

 (Worker_TP0_EP0 pid=408) INFO 04-24 10:49:25 [default_loader.py:384] Loading weights took 1292.19 seconds

 (Worker_TP0_EP0 pid=408) INFO 04-24 10:49:27 [mxfp4.py:1238] Using MoEPrepareAndFinalizeNoDPEPModular

 (Worker_TP0_EP0 pid=408) INFO 04-24 10:50:00 [gpu_model_runner.py:4848] Model loading took 74.05 GiB memory and 1307.649939 seconds

 (Worker_TP0_EP0 pid=408) INFO 04-24 10:50:20 [backends.py:1070] Using cache directory: /root/.cache/vllm/torch_compile_cache/6df291e80d/rank_0_0/backbone for vLLM's torch.compile

 (Worker_TP0_EP0 pid=408) INFO 04-24 10:50:20 [backends.py:1130] Dynamo bytecode transform time: 19.33 s [...] (Worker_TP0_EP0 pid=408) INFO 04-24 10:27:50 [mxfp4.py:481] Using 'MARLIN' Mxfp4 MoE backend.

 (Worker_TP0_EP0 pid=408) INFO 04-24 10:27:50 [deepseek_v4_attention.py:969] Using MXFP4 indexer cache for Lighening Indexer.

 (Worker_TP0_EP0 pid=408) INFO 04-24 10:27:54 [weight_utils.py:904] Filesystem type for checkpoints: 9P. Checkpoint size: 148.66 GiB. Available RAM: 0.65 GiB.

 (Worker_TP0_EP0 pid=408) INFO 04-24 10:27:54 [weight_utils.py:934] Auto-prefetch is disabled because the filesystem (9P) is not a recognized network FS (NFS/Lustre) and the checkpoint size (148.66 GiB) exceeds 90% of available RAM (0.65 GiB).

 Loading safetensors checkpoint shards: 4% Completed | 2/46 [00:33<13:38, 18.60s/it]

 Loading safetensors checkpoint shards: 7% Completed | 3/46 [00:52<13:19, 18.59s/it] [...] (EngineCore pid=273) INFO 04-24 10:27:20 [core.py:108] Initializing a V1 LLM engine (v0.1.dev15830+g8d599d76a) with config: model='/models/DeepSeek-V4-Flash', speculative_config=None, tokenizer='/models/DeepSeek-V4-Flash', skip_tokenizer_init=False, tokenizer_mode=deepseek_v4, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=1048576, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=deepseek_v4_fp8, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=fp8, device_config=cuda,

Title: Releases · vllm-project/vllm - GitHub
URL: https://github.com/vllm-project/vllm/releases
Content: 2 params (#33703), MiniMax NVFP4 weight loading (#37214), gated model HF ... FP4 kernel optimization: Up to 65% faster FP4 quantization on Blackwell

Title: GitHub - vllm-project/vllm: A high-throughput and memory-efficient ...
URL: https://github.com/vllm-project/vllm
Content: ## Contact Us

[](

   For technical questions and feature requests, please use GitHub Issues
   For discussing with fellow users, please use the vLLM Forum
   For coordinating contributions and development, please use Slack
   For security disclosures, please use GitHub's Security Advisories feature
   For collaborations and partnerships, please contact us at collaboration@vllm.ai

## Media Kit

[](

   If you wish to use vLLM's logo, please refer to our media kit repo

## About

A high-throughput and memory-efficient inference and serving engine for LLMs

vllm.ai

### Topics

amdcudainferencepytorchtransformeropenaimoellamagptmodel-servingtpukimiblackwellllmllm-servingqwendeepseekdeepseek-v3qwen3gpt-oss

### Resources

Readme

### License

Apache-2.0 license

### Code of conduct [...] ## Repository files navigation

   README
   Code of conduct
   Contributing
   Apache-2.0 license
   Security

Image 4: vLLM

### Easy, fast, and cheap LLM serving for everyone


   Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
   Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
   Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
   Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
   Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models here.

## Getting Started

 or `pip`:

undefinedshell
uv pip install vllm
undefined

Or build from source for development.

Visit our documentation to learn more.

   Installation
   Quickstart
   List of Supported Models

## Contributing

[]( [...] | CONTRIBUTING.md | CONTRIBUTING.md | [[Doc] Reorganize user guide (]( "[Doc] Reorganize user guide (#18661) Signed-off-by: DarkLight1337 <tlleungac@connect.ust.hk>")#18661) Signed-off-by: DarkLight1337 <tlleungac@connect.ust.hk>") | May 24, 2025 |
| DCO | DCO | [[Doc] Add the DCO to CONTRIBUTING.md (]( "[Doc] Add the DCO to CONTRIBUTING.md (#9803) Signed-off-by: Russell Bryant <rbryant@redhat.com> Co-authored-by: Michael Goin <michael@neuralmagic.com> Co-authored-by: Cyrus Leung <cyrus.tl.leung@gmail.com>")#9803) Signed-off-by: Russell Bryant <rbryant@redhat.com> Co-authored-by: Michael Goin <michael@neuralmagic.com> Co-authored-by: Cyrus Leung <cyrus.tl.leung@gmail.com>") | Oct 30, 2024 |
| LICENSE | LICENSE | Add Apache-2.0 license (")#102)") | May 15, 2023 |
```

---

## Assistant (Build · DeepSeek V4 Pro · 15.8s)

_Thinking:_

Important findings from the search results:

1. The upstream file is NOT at `main` branch - the 404 tells us it's in a different branch or the file path has changed. Let me try different paths.

2. Key finding: The quantization method is `deepseek_v4_fp8` (not just `fp8` or `modelopt`):
   ```
   quantization=deepseek_v4_fp8
   ```
   This is a custom quantization method specifically for DeepSeek V4!

3. The model uses `mxfp4.py` for MoE loading:
   ```
   Using 'MARLIN' Mxfp4 MoE backend.
   ```
   And the MXFP4 indexer cache:
   ```
   Using MXFP4 indexer cache for Lighening Indexer.
   ```

4. Model loading took 74.05 GiB memory for DeepSeek-V4-Flash.

So the V4 model uses a custom quantization method called `deepseek_v4_fp8` that likely combines FP8 for dense layers and MXFP4 for MoE experts. The upstream implementation has:
- A custom quant config for this mixed precision
- Uses the existing MXFP4 MoE infrastructure (MARLIN backend)
- MXFP4 indexer cache support

Let me try to find the upstream files with the correct branch/path.

**Tool: tavily_tavily_search**

**Input:**
```json
{
  "query": "site:github.com/vllm-project/vllm \"deepseek_v4_fp8\" quantization config class",
  "search_depth": "advanced"
}
```

**Output:**
```
Detailed Results:

Title: [Bug]: Quantization method specified in the model config (fp8) does not match the quantization method specified in the `quantization` argument (gguf). · Issue #19050 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/19050
Content: INFO 06-03 11:53:24 [cli\_args.py:300] non-default args: {'host': '0.0.0.0', 'port': 6011, 'tokenizer': '/data2/jcxy/llm\_model/DeepSeek-R1-0528-GGUF-UD-IQ2\_XXS', 'trust\_remote\_code': True, 'hf\_config\_path': '/data2/jcxy/llm\_model/DeepSeek-R1-0528-GGUF-UD-IQ2\_XXS', 'max\_model\_len': 8192, 'served\_model\_name': ['DeepSeek-R1-0528'], 'tensor\_parallel\_size': 4, 'gpu\_memory\_utilization': 0.98, 'enable\_prefix\_caching': True}  
INFO 06-03 11:53:24 [config.py:213] Replacing legacy 'type' key with 'rope\_type'  
INFO 06-03 11:53:32 [config.py:793] This model supports multiple tasks: {'reward', 'generate', 'embed', 'score', 'classify'}. Defaulting to 'generate'.  
Traceback (most recent call last):  
File "/data/jcxy/haolu/anaconda3/envs/haolu/bin/vllm", line 8, in [...] ## Description

@Minami-su

### Your current environment

MODEL\_PATH="/data2/jcxy/llm\_model/DeepSeek-R1-0528-GGUF-UD-IQ2\_XXS/DeepSeek-R1-0528-UD-IQ2\_XXS-00001-of-00005.gguf"  
LOG\_FILE="vllm.log"

export VLLM\_USE\_V1=0  
SERVED\_MODEL\_NAME="DeepSeek-R1-0528"  
export CUDA\_VISIBLE\_DEVICES=2,3,4,5

# 运行命令

## nohup vllm serve "$MODEL\_PATH" --hf-config-path /data2/jcxy/llm\_model/DeepSeek-R1-0528-GGUF-UD-IQ2\_XXS --tokenizer /data2/jcxy/llm\_model/DeepSeek-R1-0528-GGUF-UD-IQ2\_XXS --served-model-name "$SERVED\_MODEL\_NAME" --trust-remote-code --port 6011 --host 0.0.0.0 --dtype auto --max-model-len 8192 --gpu\_memory\_utilization 0.98 --tensor\_parallel\_size 4 --enable-prefix-caching >"$LOG\_FILE" 2>&1 &

### 🐛 Describe the bug [...] `VLLM_PLUGINS`
`quantization`

### Before submitting a new issue...

## Metadata

## Metadata

### Assignees

### Labels

### Type

### Projects

### Milestone

### Relationships

### Development

## Issue actions

## Footer

### Footer navigation

Title: [RFC]: Intel Quantization Support Roadmap (H1 2026) · Issue #37979 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/37979
Content: ### Goals

`wNa16 INT`
`w8a16 FP8`
`quant_method`
`INCConfig`

## 1. Extend Quantization Scheme Coverage

Expand Intel platform support for the quantization schemes needed by production workloads.

### 1a. wNa16 (INT) — Weight-Only Integer Quantization

### 1b. w8a16 (FP8) — FP8 Weight-Only Quantization

Note: Some schemes may depend on kernel readiness.

## 2. Architectural Cleanup

Decouple the `quant_method` dispatch logic from `INCConfig`. Today, `INCConfig.get_quant_method()` contains per-backend routing that duplicates logic already in GPTQ/AWQ/Marlin configs: [...] `AutoRoundQuantLinearMethod`
`AutoRoundMoEMethod`
`LinearMethodBase`
`FusedMoEMethodBase`
`get_method()`
`get_moe_method()`

Level 2 — Scheme-specific impl (e.g. `AutoRoundWNA16LinearImpl`, `AutoRoundFP8LinearImpl`):  
Implements an abstract `AutoRoundQuantImpl` base class that defines `create_weights()`, `process_weights_after_loading()`, and `apply_weights()`. Each impl owns a single quantization scheme and its kernel calls. [...] `AutoRoundWNA16LinearImpl`
`AutoRoundFP8LinearImpl`
`AutoRoundQuantImpl`
`create_weights()`
`process_weights_after_loading()`
`apply_weights()`
`INCConfig.get_quant_method(layer, prefix)
│
├── LinearBase → AutoRoundQuantLinearMethod.get_method(config, layer, prefix)
│ ├── wNa16 INT → AutoRoundWNA16LinearImpl
│ └── w8a16 FP8 → AutoRoundFP8LinearImpl
│
└── FusedMoE → AutoRoundMoEMethod.get_moe_method(config, layer, prefix)
├── wNa16 INT → AutoRoundWNA16MoEImpl
└── w8a16 FP8 → AutoRoundFP8MoEImpl`

### Feedback Period.

Please comment on the proposal or suggest alternatives. If there are no strong objections, we will proceed with the timeline above and submit implementation PRs. Thanks!

### CC List.

Title: [RFC]: online quantization user facing API · Issue #32412 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/32412
Content: ### Proposed Change

tl;dr;

`LLM(..., quantization="fp8_blockwise", ...)`
`LinearBase`
`FusedMoE`

#### New "online" quantization frontend

#### Observability

We should make the `VLLM_LOG_MODEL_INSPECTION=1` setting print the important details about how each layer is quantized: dtype recipe, scaling recipe.

`VLLM_LOG_MODEL_INSPECTION=1`

#### Implementation

`vllm/model_executor/layers/quantization/online`
`OnlineQuantizationConfig`

### Feedback Period.

one to two weeks

### CC List.

No response

### Any Other Things.

No response

### Before submitting a new issue...

## Metadata

## Metadata

### Assignees

### Labels

### Type

### Projects

### Milestone

### Relationships

### Development

## Issue actions

## Footer

### Footer navigation [...] @vkuzo

## Description

@vkuzo

### Motivation.

Online quantization (weights passed to vLLM in high precision, quantization of weights done inside of vLLM) is emerging as an important use case for quick experimentation and RL. Today vLLM supports online quantization with a single recipe (float8 per-tensor scaling). Let's align how to extend the user API for specifying other online quant recipes.

Today, the main user API for fp8 online quant is --quantization="fp8", which defaults to float8 tensorwise scaling for all linear and moe layers. There is no ability to specify scaling type, no ability to ignore layers, no ability to specify other dtypes for weights or activations:

See #32029 for more context on online quantization roadmap

### Proposed Change

tl;dr; [...] ## Navigation Menu

# Search code, repositories, users, issues, pull requests...

# Provide feedback

We read every piece of feedback, and take your input very seriously.

# Saved searches

## Use saved searches to filter your results more quickly

To see all available qualifiers, see our documentation.

### Uh oh!

There was an error while loading. Please reload this page.

There was an error while loading. Please reload this page.

# [RFC]: online quantization user facing API #32412

@vkuzo

## Description

@vkuzo

### Motivation.

Title: vllm/vllm/model_executor/layers/quantization/__init__.py at main · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py
Content: ## Navigation Menu

# Search code, repositories, users, issues, pull requests...

# Provide feedback

We read every piece of feedback, and take your input very seriously.

# Saved searches

## Use saved searches to filter your results more quickly

To see all available qualifiers, see our documentation.

### Uh oh!

There was an error while loading. Please reload this page.

There was an error while loading. Please reload this page.

## FilesExpand file tree

## Breadcrumbs

# \_\_init\_\_.py

## Latest commit

## History

## Breadcrumbs

# \_\_init\_\_.py

## File metadata and controls

## Footer

### Footer navigation

Title: [Feature]: Support out-of-tree register of customize quantization method · Issue #11926 · vllm-project/vllm · GitHub
URL: https://github.com/vllm-project/vllm/issues/11926
Content: ## Navigation Menu

# Search code, repositories, users, issues, pull requests...

# Provide feedback

We read every piece of feedback, and take your input very seriously.

# Saved searches

## Use saved searches to filter your results more quickly

To see all available qualifiers, see our documentation.

### Uh oh!

There was an error while loading. Please reload this page.

There was an error while loading. Please reload this page.

# [Feature]: Support out-of-tree register of customize quantization method #11926

@ice-tong

## Description

@ice-tong

### 🚀 The feature, motivation and pitch

Thanks for such excellent work! [...] Thanks for such excellent work!

We hope to leverage the fast inference capabilities of vllm to evaluate the accuracy of the quantized models. However, vllm currently does not support custom quantization schemes, and we would appreciate it if the functionality to register custom quantization schemes could be provided.

The usage would be as follows:

### Alternatives

No response

### Additional context

No response

### Before submitting a new issue...

## Metadata

## Metadata

### Assignees

### Labels

### Type

### Projects

### Milestone

### Relationships

### Development

## Issue actions

## Footer

### Footer navigation
```

**Tool: webfetch**

**Input:**
```json
{
  "format": "markdown",
  "url": "https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/vllm/model_executor/models/deepseek_v4.py",
  "max_length": 80000
}
```

**Error:**
```
StatusCode: non 2xx status code (404 GET https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/vllm/model_executor/models/deepseek_v4.py)
```

---

## Assistant (Build · DeepSeek V4 Pro · 12.4s)

_Thinking:_

The 404 is interesting. The upstream vLLM might not have the file in main branch, or it's organized differently. Let me look at the vllm-project/vllm GitHub tree to find the right path, or try to fetch it from a known commit/PR.

Looking at the issue #40778, the error mentions:
```
File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py", line 833
```

So the file definitely exists in the installed package, but might not be in the `main` branch's tree at that path. Let me try the GitHub tree view or different approaches.

Actually, looking at the vllm blog post, it says PR #40760. Let me try fetching from that PR or the commit that merged it.

Wait, the file might just be very large. Let me try fetching from GitHub's blob view instead.

**Tool: webfetch**

**Input:**
```json
{
  "format": "markdown",
  "url": "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v4.py",
  "max_length": 80000
}
```

**Error:**
```
StatusCode: non 2xx status code (404 GET https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v4.py)
```

**Tool: webfetch**

**Input:**
```json
{
  "format": "markdown",
  "url": "https://github.com/vllm-project/vllm/pull/40760/files",
  "max_length": 5000
}
```

**Output:**
```
                      \[New Model\] Support DeepseekV4 by zyongye · Pull Request #40760 · vllm-project/vllm · GitHub                                              

[Skip to content](#start-of-content)   

## Navigation Menu

Toggle navigation

[](/)

[Sign in](/login?return_to=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm%2Fpull%2F40760%2Ffiles)

Appearance settings

  

-   Platform
    
    -   AI CODE CREATION
        
        -   [
            
            GitHub CopilotWrite better code with AI
            
            ](https://github.com/features/copilot)
        -   [
            
            GitHub SparkBuild and deploy intelligent apps
            
            ](https://github.com/features/spark)
        -   [
            
            GitHub ModelsManage and compare prompts
            
            ](https://github.com/features/models)
        -   [
            
            MCP RegistryNewIntegrate external tools
            
            ](https://github.com/mcp)
        
    -   DEVELOPER WORKFLOWS
        
        -   [
            
            ActionsAutomate any workflow
            
            ](https://github.com/features/actions)
        -   [
            
            CodespacesInstant dev environments
            
            ](https://github.com/features/codespaces)
        -   [
            
            IssuesPlan and track work
            
            ](https://github.com/features/issues)
        -   [
            
            Code ReviewManage code changes
            
            ](https://github.com/features/code-review)
        
    -   APPLICATION SECURITY
        
        -   [
            
            GitHub Advanced SecurityFind and fix vulnerabilities
            
            ](https://github.com/security/advanced-security)
        -   [
            
            Code securitySecure your code as you build
            
            ](https://github.com/security/advanced-security/code-security)
        -   [
            
            Secret protectionStop leaks before they start
            
            ](https://github.com/security/advanced-security/secret-protection)
        
    -   EXPLORE
        
        -   [Why GitHub](https://github.com/why-github)
        -   [Documentation](https://docs.github.com)
        -   [Blog](https://github.blog)
        -   [Changelog](https://github.blog/changelog)
        -   [Marketplace](https://github.com/marketplace)
        
    
    [View all features](https://github.com/features)
    
-   Solutions
    
    -   BY COMPANY SIZE
        
        -   [Enterprises](https://github.com/enterprise)
        -   [Small and medium teams](https://github.com/team)
        -   [Startups](https://github.com/enterprise/startups)
        -   [Nonprofits](https://github.com/solutions/industry/nonprofits)
        
    -   BY USE CASE
        
        -   [App Modernization](https://github.com/solutions/use-case/app-modernization)
        -   [DevSecOps](https://github.com/solutions/use-case/devsecops)
        -   [DevOps](https://github.com/solutions/use-case/devops)
        -   [CI/CD](https://github.com/solutions/use-case/ci-cd)
        -   [View all use cases](https://github.com/solutions/use-case)
        
    -   BY INDUSTRY
        
        -   [Healthcare](https://github.com/solutions/industry/healthcare)
        -   [Financial services](https://github.com/solutions/industry/financial-services)
        -   [Manufacturing](https://github.com/solutions/industry/manufacturing)
        -   [Government](https://github.com/solutions/industry/government)
        -   [View all industries](https://github.com/solutions/industry)
        
    
    [View all solutions](https://github.com/solutions)
    
-   Resources
    
    -   EXPLORE BY TOPIC
        
        -   [AI](https://github.com/resources/articles?topic=ai)
        -   [Software Development](https://github.com/resources/articles?topic=software-development)
        -   [DevOps](https://github.com/resources/articles?topic=devops)
        -   [Security](https://github.com/resources/articles?topic=security)
        -   [View all topics](https://github.com/resources/articles)
        
    -   EXPLORE BY TYPE
        
        -   [Customer stories](https://github.com/customer-stories)
        -   [Events & webinars](https://github.com/resources/events)
        -   [Ebooks & reports](https://github.com/resources/whitepapers)
        -   [Business insights](https://github.com/solutions/executive-insights)
        -   [GitHub Skills](https://skills.github.com)
        
    -   SUPPORT & SERVICES
        
        -   [Documentation](https://docs.github.com)
        -   [Customer support](https://support.github.com)
        -   [Community forum](https://github.com/orgs/community/discussions)
        -   [Trust center](https://github.com/trust-center)
        -   [Partners](https://github.com/partners)
        
    
    [View all resources](https://github.com/resources)
    
-   Open Source
    
    -   COMMUNITY
        
        -   [
            
            GitHub SponsorsFund open source developers
            
            ](https://github.com/sponsors)
        
    -   PROGRAMS
        
        -   [Security Lab](https://securitylab.github.com)
        -   [Maintainer Community](https://maintainers.github.com)
        -   [Accelerator](https://github.com/accelerator)
        -   [GitHub Stars](https://stars.github.com)
        -   [Archive Program](https://archiveprogram.github.com)
        
    -   REPOSITORIES
        
        -   [Topics](https://github.com/topics)
        -   [Trending](https://github.com/trending)
        -   [Collections](https://github.com/collections)
        
    
-   Enterprise
    
    -   ENTERPRISE SOLUTIONS
        
        -   [
            
            Enterprise platformAI-powered developer platform
            
            ](https://github.com/enterprise)
        
    -   AVAILABLE ADD-ONS
        
        -   [
            
            GitHub Advanced SecurityEnterprise-grade security features
            
            ](https://github.com/security/advanced-security)
        -   [
            
            Copilot for BusinessEnterprise-grade AI features
            
            ](https://github.com/features/copilot/copilot-business)
        -   [
            
            Premium SupportEnterprise-grade 24/7 support
            
            ](https://github.com/premium-support)
        
    
-   [Pricing](https://github.com/pricing)

Search or jump to...

# Search code, repositories, users, issues, pull requests...

Search

Clear

[Search syntax tips](https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax)

# Provide feedback

We read every piece of feedback, and take your input very seriously.

 Include my email address so I can be contacted

Cancel Submit feedback

# Saved searches

## Use saved searches to filter your results more quickly

Name  

Query 

To see all available qualifiers, see our [documentation](https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax).

Cancel Create saved search

[Sign in](/login?return_to=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm%2Fpull%2F40760%2Ffiles)

[Sign up](/signup?ref_cta=Sign+up&ref_loc=header+logged+out&ref_page=%2F%3Cuser-name%3E%2F%3Crepo-name%3E%2Fpull_requests%2Fshow%2Ffiles&source=header-repo&source_repo=vllm-project%2Fvllm)

Appearance settings

Resetting focus

You signed in with another tab or window. Reload to refresh your session. You signed out in another tab or window. Reload to refresh your session. You switched accounts on another tab or window. Reload to refresh your session. Dismiss alert

[vllm-project](/vllm-project) / **[vllm](/vllm-project/vllm)** Public

-   ### Uh oh!
    
    There was an error while loading. Please reload this page.
    
-   [Notifications](/login?return_to=%2Fvllm-project%2Fvllm) You must be signed in to change notification settings
-   [Fork 16.1k](/login?return_to=%2Fvllm-project%2Fvllm)
-   [Star 78.2k](/login?return_to=%2Fvllm-project%2Fvllm)
    

-   [Code](/vllm-project/vllm)
-   [Issues 1.9k](/vllm-project/vllm/issues)
-   [Pull requests 2.6k](/vllm-project/vllm/pulls)
-   [Discussions](/vllm-project/vllm/discussions)
-   [Actions](/vllm-project/vllm/actions)
-   [Projects](/vllm-project/vllm/projects)
-   [Security and quality 39](/vllm-project/vllm/security)
-   [Insights](/vllm-project/vllm/pulse)

Additional navigation options

-   [Code](/vllm-project/vllm)
-   [Issues](/vllm-project/vllm/issues)
-   [Pull requests](/vllm-project/vllm/pulls)
-   [Discussions](/vllm-project/vllm/discussions)
-   [Actions](/vllm-project/vllm/actions)
-   [Projects](/vllm-project/vllm/projects)
-   [Security and quality](/vllm-project/vllm/security)
-   [Insights](/vllm-project/vllm/pulse)

# \[New Model\] Support DeepseekV4 #40760

New issue

**Have a question about this project?** Sign up for a free GitHub account to open an issue and contact its maintainers and the community.

[Sign up for GitHub](/signup?return_to=%2Fvllm-project%2Fvllm%2Fissues%2Fnew%2Fchoose)

By clicking “Sign up for GitHub”, you agree to our [terms of service](https://docs.github.com/terms) and [privacy statement](https://docs.github.com/privacy). We’ll occasionally send you account related emails.

Already on GitHub? [Sign in](/login?return_to=%2Fvllm-project%2Fvllm%2Fissues%2Fnew%2Fchoose) to your account

Open

[zyongye](/zyongye) wants to merge 10 commits into [vllm-project:main](/vllm-project/vllm/tree/main "vllm-project/vllm:main")

*base:* main

Choose a base branch

Branches Tags

Loading

Loading

 

from [zyongye:dsv4](/zyongye/vllm/tree/dsv4 "zyongye/vllm:dsv4")  

[Conversation 31](/vllm-project/vllm/pull/40760) [Commits 10](/vllm-project/vllm/pull/40760/commits) [Checks 3](/vllm-project/vllm/pull/40760/checks) [Files changed](/vllm-project/vllm/pull/40760/files)

### Uh oh!

There was an error while loading. Please reload this page.

Open

# [\[New Model\] Support DeepseekV4](#top) #40760

Show file tree

Hide file tree

Changes from **all commits**

Commits

[

Show all changes

10 commits

](/vllm-project/vllm/pull/40760/files)

Select commit Hold shift + click to select a range

[

`434b1bd`

SVF support squashed

zyongye Jan 7, 2026

](/vllm-project/vllm/pull/40760/commits/434b1bdb67515a07843dedb59477f681ff326e46)[

`bc34b25`

nit

zyongye Apr 24, 2026

](/vllm-project/vllm/pull/40760/commits/bc34b25e2055a69526dcb1c454d863049fb4ebfc)[

`4bab47b`

\[Fix\] Always allocate FP8 indexer cache even for FP4 indexer (#225)

ivanium Apr 24, 2026

](/vllm-project/vllm/pull/40760/commits/4bab47b4396b073246e0aed35e4f34043a7e8906)[

`6f3820e`

Avoid CPU-GPU sync (#224)

WoosukKwon Apr 24, 2026

](/vllm-project/vllm/pull/40760/commits/6f3820e838a1fa4776cc0aab97c0dad0f1223063)[

`aa11460`

add 1024 topk support (#226)

zyongye Apr 24, 2026

](/vllm-project/vllm/pull/40760/commits/aa114601db92d17180a8cd382346a290ed5ff71b)[

`3602f14`

temporary disable persistent topk for 1024

zyongye Apr 24, 2026

](/vllm-project/vllm/pull/40760/commits/3602f14f0e146b234be911d916e381b4e6a4dc0c)[

`4258ac3`

Integrate MegaMoE

WoosukKwon Apr 25, 2026

](/vllm-project/vllm/pull/40760/commits/4258ac345584f8179ed4881c2fdd4b4df0b4a55e)[

`06e4b4f`

Add model change

WoosukKwon Apr 25, 2026

](/vllm-project/vllm/pull/40760/commits/06e4b4f5b2063193c3d390f4423d6e1c26403f46)[

`6d244bd`

Support dummy loading

WoosukKwon Apr 25, 2026

](/vllm-project/vllm/pull/40760/commits/6d244bdb4d9023a1d2fac1981d488c477ea263ec)[

`e8e38e1`

free up unused weights

WoosukKwon Apr 25, 2026

](/vllm-project/vllm/pull/40760/commits/e8e38e1686c3ca0835b9556fc1f9b28b9e1a455f)

**File filter**

### Filter by extension

Filter by extension  .cmake  (2)  .cpp  (2)  .cu  (7)  .cuh  (1)  .h  (2)  .json  (5)  .md  (1)  .mooncake-layer  (1)  .nightly\_torch  (1)  .py  (127)  .sh  (1)  .txt  (7)  No extension  (1)  All 13 file types selected

---

 Only manifest files

---

 Viewed files

[Clear filters](/vllm-project/vllm/pull/40760/files)

**Conversations**

Failed to load comments. Retry

Loading

### Uh oh!

There was an error while loading. Please reload this page.

**Jump to**

Jump to file

Failed to load files. Retry

Loading

### Uh oh!

There was an error while loading. Please reload this page.

##### Diff view

 ![Unified Diff View](https://github.githubassets.com/assets/unified-6de447b07fd7.svg)

 Unified

 ![Split Diff View](https://github.githubassets.com/assets/split-b930d4a1df45.svg)

 Split

  Hide whitespace

Apply and reload

 Show whitespace

##### Diff view

 ![Unified Diff View](https://github.githubassets.com/assets/unified-6de447b07fd7.svg)

 Unified

 ![Split Diff View](https://github.githubassets.com/assets/split-b930d4a1df45.svg)

 Split

  Hide whitespace

Apply and reload

-   CMakeLists.txt [CMakeLists.txt](#diff-1e7de1ae2d059d21e1dd75d5812d5a34b0222cef273b7c3a2af62eb747f9d20a)
-   benchmarks/kernels
    -   benchmarks/kernels/bench\_compressor\_fusion.py [bench\_compressor\_fusion.py](#diff-98e94abda05351bf0a8400d81e53a1723ae09810e0377553ca3e9360a9c44edc)
-   cmake/external\_projects
    -   cmake/external\_projects/deepgemm.cmake [deepgemm.cmake](#diff-452d607ec370ad197b9eeba5c89bb4f7d44ebb6cbe83b0b9f3ff7d81c46811d2)
    -   cmake/external\_projects/flashmla.cmake [flashmla.cmake](#diff-660889ccd173b5116fd8d8a3444823f5cbb201922c236ca5932cde917efba40d)
-   csrc
    -   csrc/cache\_kernels.cu [cache\_kernels.cu](#diff-de25aaedde5971ab468ed0ec02ea3f4e4dde21580bc858f2f8640b31cd566891)
    -   csrc/fused\_deepseek\_v4\_qnorm\_rope\_kv\_insert\_kernel.cu [fused\_deepseek\_v4\_qnorm\_rope\_kv\_insert\_kernel.cu](#diff-c2d275709ce15a27c9af43cf06858ef5d7e84d64252017461499bc08aca8276f)
    -   csrc/layernorm\_kernels.cu [layernorm\_kernels.cu](#diff-8f7b2a2446ba48c6c4e625dc3348c75a9fef66ab904b143fdbf129efc5b0e236)
    -   moe
        -   csrc/moe/moe\_ops.h [moe\_ops.h](#diff-09bcbc1f33b35d031150fdfbd967fb530b68b19bc029e83054d1d9614f151ece)
        -   csrc/moe/topk\_softplus\_sqrt\_kernels.cu [topk\_softplus\_sqrt\_kernels.cu](#diff-a1d46d48582d71db09175612ec0a02d313a024b4e6f9b2a2e0c7394aa2e1ff48)
        -   csrc/moe/torch\_bindings.cpp [torch\_bindings.cpp](#diff-c61daf2bf1e2fac9804f061aabeb084d9f37dafac95ecbbba1fbec853eb9d444)
    -   csrc/ops.h [ops.h](#diff-02fe44d8075bde8441e12d4824c1cc6a83dd75e27f882e10c73d16ad47d4bed8)
    -   csrc/persistent\_topk.cuh [persistent\_topk.cuh](#diff-148f73d59829919f3d61c02c30e1f9f7083c73ec85b39e3fabc918bfcc91e47d)
    -   csrc/pos\_encoding\_kernels.cu [pos\_encoding\_kernels.cu](#diff-77cc09d6c2d0e42847236af3c8702f40c23d9921e59db2e98f346edfce58c145)
    -   csrc/sampler.cu [sampler.cu](#diff-3aa9e99af7cc5aca5143b440b6092c0a2537f1f0600a71ec75421b29583d5737)
    -   csrc/topk.cu [topk.cu](#diff-8fa59dac78706378bd1c9f10953cc26b98583920338832659411f997e48af75b)
    -   csrc/torch\_bindings.cpp [torch\_bindings.cpp](#diff-083f046c1bdee081e6305e4f7cc4801117ef83fc8bf5d14374754045fb1e437c)
-   docker
    -   docker/Dockerfile [Dockerfile](#diff-f34da55ca08f1a30591d8b0b3e885bcc678537b2a9a4aadea4f190806b374ddc)
    -   docker/Dockerfile.mooncake-layer [Dockerfile.mooncake-layer](#diff-c30b80d8b265524aa1432c55a878128b87835a45a881ac23cdf9c966ec6db356)
    -   docker/Dockerfile.nightly\_torch [Dockerfile.nightly\_torch](#diff-24be9d500a470891f130dc6ca22e9268889f25db8c24acbea8e3a7468ac454d0)
    -   docker/versions.json [versions.json](#diff-d771268f20137703e8ff8230c727fdc8c6ef784975cde0e8f6f744e0b0b74b41)
-   docs/design
    -   docs/design/attention\_backends.md [attention\_backends.md](#diff-09edded74914021771d3d06ad567e2c796ac78906006d344531db3cc50ab95f8)
-   requirements
    -   requirements/cuda.txt [cuda.txt](#diff-f0bb597ba1276975a33aef58f30237934c3fa54f627f5020210df7301acba61a)
    -   requirements/kv\_connectors.txt [kv\_connectors.txt](#diff-3563710e4a9a64ed34b15979dc95e4cc8dcb9b72933736de661d5c9a8550f192)
-   tests
    -   kernels
        -   attention
            -   tests/kernels/attention/test\_use\_trtllm\_attention.py [test\_use\_trtllm\_attention.py](#diff-7af0fc5bfc6ffa9a077d7412c604f5caf35595a51ef0e63ad6c063c0a021607f)
        -   core
            -   tests/kernels/core/test\_fused\_q\_kv\_rmsnorm.py [test\_fused\_q\_kv\_rmsnorm.py](#diff-14be24b1053112ab57e987e83abaaea95d9cecd6b5ec2b7d745beabc0a4091db)
        -   moe
            -   tests/kernels/moe/test\_deepgemm.py [test\_deepgemm.py](#diff-2aa120ce7378412f95fa5ca4010dc8606d8c3e717d66abfd3c30fa20f5d28b74)
            -   tests/kernels/moe/test\_ocp\_mx\_moe.py [test\_ocp\_mx\_moe.py](#diff-9ac2edd8955e86ce70c891d58c378f39b06d13d09aa7d621747b8788376bb200)
            -   tests/kernels/moe/test\_topk\_softplus\_sqrt.py [test\_topk\_softplus\_sqrt.py](#diff-4371824f87c86c93cf45940a144d5c21a74ed73edcedcf070298164e100565e8)
        -   tests/kernels/test\_compressor\_kv\_cache.py [test\_compressor\_kv\_cache.py](#diff-c71bb02137e93fc3063ff93fe596fc1819055c9517a865577906df6455ce9d9d)
        -   tests/kernels/test\_fused\_deepseek\_v4\_qnorm\_rope\_kv\_insert.py [test\_fused\_deepseek\_v4\_qnorm\_rope\_kv\_insert.py](#diff-b6192ed11f99c0c066f4adaa763403df2c6108681749aa0ff5b925d62f874be5)
        -   tests/kernels/test\_fused\_indexer\_q\_rope\_quant.py [test\_fused\_indexer\_q\_rope\_quant.py](#diff-52b3ff09e55be94c6750f3d7c84bd500e7816b05a7d9941da7b2f70822ee1379)
        -   tests/kernels/test\_fused\_inv\_rope\_fp8\_quant.py [test\_fused\_inv\_rope\_fp8\_quant.py](#diff-570892970eb2fdb0bf8ee286bebbbd8827ba624c05e4a45d7318b91a4f75da87)
        -   tests/kernels/test\_top\_k\_per\_row.py [test\_top\_k\_per\_row.py](#diff-848efcc44dab0e92bb2d35e9aea02e56e3a3e39d21192f21ca7272543ddb996d)
    -   model\_executor
        -   tests/model\_executor/test\_routed\_experts\_capture.py [test\_routed\_experts\_capture.py](#diff-bd1370684b09c20a481bf4923c6411f82d99e2c5dffb50cc177746bd8c4aced5)
    -   models
        -   tests/models/registry.py [registry.py](#diff-c2cd72327248d1c1aa3d4b29ec9e47314d9893bfeff94e927841cd640fac84c1)
        -   tests/models/test\_deepseek\_v4\_mega\_moe.py [test\_deepseek\_v4\_mega\_moe.py](#diff-a68e6770c29578f1c4bb68e7680751a64be6ce91a2f301569b7ec660a72e244f)
    -   reasoning
        -   tests/reasoning/test\_deepseekv3\_reasoning\_parser.py [test\_deepseekv3\_reasoning\_parser.py](#diff-be88340004f01fa4d163632888ada1577028d665eaa57fddeeeeeca4b1f2f4e0)
    -   tokenizers\_
        -   fixtures/deepseek\_v4
            -   tests/tokenizers\_/fixtures/deepseek\_v4/test\_input\_1.json [test\_input\_1.json](#diff-6a2227e359f3f56a5f55f3422a8f4d3f1a5f5faf5a650978cf471f05e184c5ce)
            -   tests/tokenizers\_/fixtures/deepseek\_v4/test\_input\_2.json [test\_input\_2.json](#diff-d980472a4ce0a71ad075998a5292c21593e7dcfe3ebec33b929a845fd30abfe0)
            -   tests/tokenizers\_/fixtures/deepseek\_v4/test\_input\_3.json [test\_input\_3.json](#diff-784da55bcca8b34a98670dc984748da6dc872aa84a0925e4d7259b19137307f8)
            -   tests/tokenizers\_/fixtures/deepseek\_v4/test\_input\_4.json [test\_input\_4.json](#diff-b277ee6ca252bed01ad508c3a3fa9af26eb07a8c868a9ff4126f93b2b410d653)
            -   tests/tokenizers\_/fixtures/deepseek\_v4/test\_output\_1.txt [test\_output\_1.txt](#diff-95e8853f41039296c7a8bf8e9d85f78b214e36c866854d5beee30cf6baf602da)
            -   tests/tokenizers\_/fixtures/deepseek\_v4/test\_output\_2.txt [test\_output\_2.txt](#diff-2bb953eec4f08ffaa3de132a7099a411c00629a9c2b47bd6fcbef8781e579c74)
            -   tests/tokenizers\_/fixtures/deepseek\_v4/test\_output\_3.txt [test\_output\_3.txt](#diff-48d473bf1efc6bed4116204d6da706bbd32a43a197ca39012ec9cfdf6d753dce)
            -   tests/tokenizers\_/fixtures/deepseek\_v4/test\_output\_4.txt [test\_output\_4.txt](#diff-04e4af13770f1633308855ee8d361b28e6fe7bb39dcdb7ab0a0954321ce0add2)
        -   tests/tokenizers\_/test\_deepseek\_v4.py [test\_deepseek\_v4.py](#diff-5b28b777cebaa10432c10ba88ffc30934316b5ff9274de52add405cf46657fc4)
    -   tool\_parsers
        -   tests/tool\_parsers/test\_deepseekv4\_tool\_parser.py [test\_deepseekv4\_tool\_parser.py](#diff-7037d8e3116060ef2ace80d49a54456f170222db4b1841d325210ea4d7b9cccc)
    -   v1
        -   attention
            -   tests/v1/attention/test\_indexer\_deepseek\_v4\_slot\_mapping.py [test\_indexer\_deepseek\_v4\_slot\_mapping.py](#diff-7c17a99a1c7abbfd813a1039b101cf5d572b17ec1860e39467abe1cec937b58d)
        -   core
            -   tests/v1/core/test\_kv\_cache\_utils.py [test\_kv\_cache\_utils.py](#diff-1af0038ebe3c23c015eacc37a83bc7c5b8087480211685d7f3993528abb791d7)
            -   tests/v1/core/test\_prefix\_caching.py [test\_prefix\_caching.py](#diff-f52247bf62a5b0dfa11b44bdd30caa66ec923282a137afde016f8eff330137ef)
            -   tests/v1/core/test\_scheduler.py [test\_scheduler.py](#diff-2b86958c47f597da5299d97e5c1c57751f77fa4e8447c1ab0ec26307a00be471)
        -   kv\_connector/unit
            -   tests/v1/kv\_connector/unit/test\_mooncake\_connector.py [test\_mooncake\_connector.py](#diff-226a74e1a63aa275a9d0a3e08c553e78e4757a98e9d9078a4ff833f10059c008)
            -   tests/v1/kv\_connector/unit/test\_mooncake\_connector\_hma.py [test\_mooncake\_connector\_hma.py](#diff-62690c47ec452dcd3003eaa2586a5627cd8b7e376739ead07c02ffa0f7e04692)
        -   streaming\_input
            -   tests/v1/streaming\_input/test\_scheduler\_streaming.py [test\_scheduler\_streaming.py](#diff-3345a5e097213fa9eb75a0be8d296d331e7d7cfa9d3c7a2026107e8770e6d45c)
-   tools
    -   tools/install\_deepgemm.sh [install\_deepgemm.sh](#diff-76c291e9c9733800cfb0c7f9bc95903a2d13417c798dd41134d94dce38bc08b9)
-   vllm
    -   vllm/\_custom\_ops.py [\_custom\_ops.py](#diff-b2645ce390db5e2ac2123144700f912fab9458314789aa8245c657cf42c6039e)
    -   config
        -   vllm/config/attention.py [attention.py](#diff-1b15c9b9ea534e3e02c27d1138d6fdc176eeedeab43244549f63ac380e62613c)
        -   vllm/config/cache.py [cache.py](#diff-d4949a2bcea6f8703ee705f55ab36b1f0e7520c8e132d4eb9d550fdf14bba835)
        -   vllm/config/compilation.py [compilation.py](#diff-19413af30c8d2ba8cdaf4989476312ece57b0e9811b39a62286623b99c7fd0a8)
        -   vllm/config/kernel.py [kernel.py](#diff-83520694c7cbb8038d4a2216a51304e81a3fcd674cb43f4aa2d19c14b588d3cd)
        -   vllm/config/model.py [model.py](#diff-998c640befaf137b9af825f29f4e6e47d273caab1fd04093c97df24b18f5c417)
        -   vllm/config/speculative.py [speculative.py](#diff-d9f90bb1c6d4f572294cec8ce7251c46224d8f6ecf5918d7f8a9fb768bb65dbb)
    -   distributed/kv\_transfer/kv\_connector/v1/mooncake
        -   vllm/distributed/kv\_transfer/kv\_connector/v1/mooncake/mooncake\_connector.py [mooncake\_connector.py](#diff-5dda20fce12a31c71aae2220bd300f66458e0919d8cecf69d54591f7dd7365ac)
    -   vllm/envs.py [envs.py](#diff-b7538eee1327212c7cfbe3f7874600dfcd41b5fb5662695cd5f396294610ddf9)
    -   model\_executor
        -   kernels/linear/scaled\_mm
            -   vllm/model\_executor/kernels/linear/scaled\_mm/deep\_gemm.py [deep\_gemm.py](#diff-9a1a1730a1fc1868db95c85ee80170d8ab45f4db59d372a81225c6a6268cfc27)
        -   layers
            -   attention
                -   vllm/model\_executor/layers/attention/mla\_attention.py [mla\_attention.py](#diff-e74ff3870f4b147beb297d6176081f70fd2bdf58fd38b182da58307e4da9eb2a)
            -   vllm/model\_executor/layers/deepseek\_compressor.py [deepseek\_compressor.py](#diff-5410ab439b275db4ba56bb9574098619ad64d99e5531ad9ddec262622a5edb41)
            -   vllm/model\_executor/layers/deepseek\_v4\_attention.py [deepseek\_v4\_attention.py](#diff-6aaf33c1df71fb9b4678ed59bb7123902f51f853c4fa4d51bcbf2ab0e3bae067)
            -   fused\_moe
                -   vllm/model\_executor/layers/fused\_moe/config.py [config.py](#diff-3886f75aa77ead65142d91249abd431fb740df5242672c064bb8018dae15b171)
                -   experts
                    -   vllm/model\_executor/layers/fused\_moe/experts/deep\_gemm\_moe.py [deep\_gemm\_moe.py](#diff-41c566c3da38ffedab2b017fdc686d689f10270e6e233e12c43eff862b15251b)
                    -   vllm/model\_executor/layers/fused\_moe/experts/trtllm\_mxfp4\_moe.py [trtllm\_mxfp4\_moe.py](#diff-bdfb4fe1bb3692d110b05d32ada5874882a9f0433cb9d6e8b04a5f5f8346eb30)
                -   vllm/model\_executor/layers/fused\_moe/fused\_marlin\_moe.py [fused\_marlin\_moe.py](#diff-506aa9619c06a4337f98f6bfe9fb00a4a9b907b13f49c3ad7d9c9b0850a47789)
                -   vllm/model\_executor/layers/fused\_moe/fused\_moe\_method\_base.py [fused\_moe\_method\_base.py](#diff-e65897462d5896dd4ecf50b9301e6d3d5b238606bdbc8afbc667d27d6e073d75)
                -   vllm/model\_executor/layers/fused\_moe/gpt\_oss\_triton\_kernels\_moe.py [gpt\_oss\_triton\_kernels\_moe.py](#diff-766ad90bbb5ee8239a3714a0a9d24b5723fdecb6ac6a6db2bc56c245e7412199)
                -   vllm/model\_executor/layers/fused\_moe/layer.py [layer.py](#diff-eddafffeb6f159f8c75f635d18a502fcfbf662a562b1ae7a8683a9790161a10b)
                -   oracle
                    -   vllm/model\_executor/layers/fused\_moe/oracle/mxfp4.py [mxfp4.py](#diff-7626acc01163a9549f321f672f7b4d6395d555f5f3cdb9c2c3a6a9983b9a7246)
                -   router
                    -   vllm/model\_executor/layers/fused\_moe/router/base\_router.py [base\_router.py](#diff-501bac7c64cf5171cf975d399a153908297134b345688299c69fa69ce60edebe)
                    -   vllm/model\_executor/layers/fused\_moe/router/custom\_routing\_router.py [custom\_routing\_router.py](#diff-46481bf6989f9f268f73f25c2f62e0a5b33fbbdc7ed0e51e09be24304c61b28f)
                    -   vllm/model\_executor/layers/fused\_moe/router/fused\_moe\_router.py [fused\_moe\_router.py](#diff-4f8ca683e3a8ab3c4be6e6347db9b18332a0fdfe44f601c3c8585645dfad3640)
                    -   vllm/model\_executor/layers/fused\_moe/router/fused\_topk\_bias\_router.py [fused\_topk\_bias\_router.py](#diff-f4ad537ccb95db48af2470eb531dd6933e0931bd90663599491435a81a3ab720)
                    -   vllm/model\_executor/layers/fused\_moe/router/fused\_topk\_router.py [fused\_topk\_router.py](#diff-bb09392cf37edd7a75a69848377e061bffa7318c30858de52b4bf9e4ba8ff55c)
                    -   vllm/model\_executor/layers/fused\_moe/router/grouped\_topk\_router.py [grouped\_topk\_router.py](#diff-b89dc19b95122f28b1c2f76820a8a0129ce06485948a155696e4b826e7d1312b)
                    -   vllm/model\_executor/layers/fused\_moe/router/router\_factory.py [router\_factory.py](#diff-272b8da51e1bcd7f6bef1c3db1e44ffa4d2c4db9259f9c48d313a81d6531e02b)
                    -   vllm/model\_executor/layers/fused\_moe/router/routing\_simulator\_router.py [routing\_simulator\_router.py](#diff-f60bbda5b8e69ed8d3eb8b7a3475a8f1d0ae1e56790b33c88a56c19ac2a5b093)
                -   runner
                    -   vllm/model\_executor/layers/fused\_moe/runner/chunking\_moe\_runner.py [chunking\_moe\_runner.py](#diff-ec97b5f362f2090c1fc30aac595cbff4fb862dfa8362acc134cb1227c21c9b51)
                    -   vllm/model\_executor/layers/fused\_moe/runner/default\_moe\_runner.py [default\_moe\_runner.py](#diff-07b6e02f2d6d0613c245b0a93fcdfebfdac0e0631d2727f4a7f311ba8bf9d86a)
                    -   vllm/model\_executor/layers/fused\_moe/runner/moe\_runner.py [moe\_runner.py](#diff-e4f7854e5c7b372f38317b0886343a226e388c9ffd9851f55d685a6e465b1a84)
                    -   vllm/model\_executor/layers/fused\_moe/runner/moe\_runner\_base.py [moe\_runner\_base.py](#diff-21ab2e0c97e45b4e64a9a737dc3f2abb9242db3cf0b9bdbbd7b8a81fd8c3f512)
                -   vllm/model\_executor/layers/fused\_moe/shared\_fused\_moe.py [shared\_fused\_moe.py](#diff-43fde58b96a155b33f507293cabdd1d69a014c1edccb54cc54e9325b7729a5ae)
                -   vllm/model\_executor/layers/fused\_moe/unquantized\_fused\_moe\_method.py [unquantized\_fused\_moe\_method.py](#diff-c0d225629be6540e56774387bac52ce8e23c2de8fce3a8b6388b55486a06bfc0)
                -   vllm/model\_executor/layers/fused\_moe/utils.py [utils.py](#diff-3e585ddadd72f829102aacb7d5499c9cf1a4f9e6654549a77ebd03addc65846d)
            -   vllm/model\_executor/layers/mhc.py [mhc.py](#diff-41ea6d69f9c18f7826268f6e812bd4a0b2e8494bc6f356fafe9f4400b7073d78)
            -   quantization
                -   vllm/model\_executor/layers/quantization/\_\_init\_\_.py [\_\_init\_\_.py](#diff-a34a1f8688c54ba32fb385873ef424b85959a1af82561f1256c4f76ce7c2bd6b)
                -   compressed\_tensors/compressed\_tensors\_moe
                    -   vllm/model\_executor/layers/quantization/compressed\_tensors/compressed\_tensors\_moe/compressed\_tensors\_moe\_w4a4\_nvfp4.py [compressed\_tensors\_moe\_w4a4\_nvfp4.py](#diff-4dba9020d1457ca7636ba9b85610f93202b17d4cd57d09d4e0b12ff15e02d756)
                    -   vllm/model\_executor/layers/quantization/compressed\_tensors/compressed\_tensors\_moe/compressed\_tensors\_moe\_w4a8\_int8.py [compressed\_tensors\_moe\_w4a8\_int8.py](#diff-20890dd9c2ea14b73bcf7e983534d74a2cd119b06b38e9abbd6519d11aa880af)
                    -   vllm/model\_executor/layers/quantization/compressed\_tensors/compressed\_tensors\_moe/compressed\_tensors\_moe\_w8a8\_fp8.py [compressed\_tensors\_moe\_w8a8\_fp8.py](#diff-162609d295ebb7c4ccf34c5d88e9144c1eea8e258f6217f56744b52027fb3afb)
                    -   vllm/model\_executor/layers/quantization/compressed\_tensors/compressed\_tensors\_moe/compressed\_tensors\_moe\_w8a8\_mxfp8.py [compressed\_tensors\_moe\_w8a8\_mxfp8.py](#diff-b13dc105f80420fa6e1fd606e5603cf0be360da486635067b3335c4a09cf225e)
                    -   vllm/model\_executor/layers/quantization/compressed\_tensors/compressed\_tensors\_moe/compressed\_tensors\_moe\_wna16\_marlin.py [compressed\_tensors\_moe\_wna16\_marlin.py](#diff-6988d84725da155cee907fff67525711a47247ee76c0eb0b8bc0173e51048f71)
                -   vllm/model\_executor/layers/quantization/fp8.py [fp8.py](#diff-5511bfcc9c53f7d96517ad43e4087f6777bef21302da983f42cafae40a866644)
                -   vllm/model\_executor/layers/quantization/modelopt.py [modelopt.py](#diff-b214ec1163353d1515927f1e8155b83f36aeee1de9620bb057535847a6f1b3e0)
                -   vllm/model\_executor/layers/quantization/mxfp4.py [mxfp4.py](#diff-c73528091f2176f6547e24074950962deec4311315fa5e99ef2ca7682680708c)
                -   online
                    -   vllm/model\_executor/layers/quantization/online/fp8.py [fp8.py](#diff-92d8d1eab32c063cda9e5e05c00287dc96c1f914141582c74dd6cb59f664d91b)
                -   quark
                    -   vllm/model\_executor/layers/quantization/quark/quark\_moe.py [quark\_moe.py](#diff-81df2ed125290511a8464e44614dc9a3701cc089b2fe38abd2f19d8269b30eb6)
                -   utils
                    -   vllm/model\_executor/layers/quantization/utils/fp8\_utils.py [fp8\_utils.py](#diff-cce58c0ceb6a9b15a01f117d734b93736acc25ed89921c2eacc58ea05bd34d0e)
            -   rotary\_embedding
                -   vllm/model\_executor/layers/rotary\_embedding/\_\_init\_\_.py [\_\_init\_\_.py](#diff-4135ce0d65fd99725ceb5af3025c1090098cd5d8067531021cf5cc5d5120b6de)
                -   vllm/model\_executor/layers/rotary\_embedding/deepseek\_scaling\_rope.py [deepseek\_scaling\_rope.py](#diff-eb92572a9c15355e236585bf8c64f01363ab8d04f61385d72ae8696e8568f170)
            -   vllm/model\_executor/layers/sparse\_attn\_indexer.py [sparse\_attn\_indexer.py](#diff-3d0ef393a0b3877bff02cc97cd933ad874c8ad84c83b984009d09568ef83331f)
            -   vllm/model\_executor/layers/utils.py [utils.py](#diff-8de4b74e35fe9c8f4ebdef6017952ff0a68111b87ca29d1a5874a35670b0bda2)
        -   models
            -   vllm/model\_executor/models/config.py [config.py](#diff-3bc5cc9c6f85c41c016f7a08808cc0b3021c8e13fd7ad92aa20077efb1f4a354)
            -   vllm/model\_executor/models/deepseek\_v4.py [deepseek\_v4.py](#diff-146fe7ce9c628d1f78b8a162a856a862d1ce7ac4548d875875ab124c0ab03c0c)
            -   vllm/model\_executor/models/deepseek\_v4\_mtp.py [deepseek\_v4\_mtp.py](#diff-1063f2fda4cd30f15cf85bcea4b31d73497380b9cbacd33a130e2eb31da07dbe)
            -   vllm/model\_executor/models/registry.py [registry.py](#diff-0852f1e9753819abe4f85380abdc0660fd4a998f44bd7bb73d68f462bd776d95)
        -   offloader
            -   vllm/model\_executor/offloader/prefetch.py [prefetch.py](#diff-efedb9f48f57b7fa3edb70e3ad79ca14cdf8490bb233390c47ee644efc869df2)
    -   reasoning
        -   vllm/reasoning/\_\_init\_\_.py [\_\_init\_\_.py](#diff-980744dc54574977a5d1917e36e795bf965871d4a76bd699dcc6e7cd53f82d6a)
    -   renderers
        -   vllm/renderers/deepseek\_v4.py [deepseek\_v4.py](#diff-1403864aca61311d40d07c24e72bb08484fac8e3b3f93987907d759b0780e0f8)
        -   vllm/renderers/registry.py [registry.py](#diff-1ea7902b8feadd321bda05c64b9778dcae4be8ec08e15ad4bcc52ead6d555634)
    -   tokenizers
        -   vllm/tokenizers/deepseek\_v4.py [deepseek\_v4.py](#diff-a91dc1fddec5aaaaf8c77a6d9eaaf8f1e8951c4be810eaecb0c7ede86157d870)
        -   vllm/tokenizers/deepseek\_v4\_encoding.py [deepseek\_v4\_encoding.py](#diff-d4055ea1f14dfc37d9498bbd938e915158889226aeda758ff41e567a3955e9cb)
        -   vllm/tokenizers/registry.py [registry.py](#diff-0a0224a2416844000611f12e6fc060d0b268b2ff6432a7392b65c839a94c498d)
    -   tool\_parsers
        -   vllm/tool\_parsers/\_\_init\_\_.py [\_\_init\_\_.py](#diff-c83675ab17381b90af54d7e0b578580b783664ed283ddd878430fd81c0ad025f)
        -   vllm/tool\_parsers/deepseekv32\_tool\_parser.py [deepseekv32\_tool\_parser.py](#diff-4ce15d2337946ef9ffdba3d897dec72a5a1928a7a9153b85b3e623e31eefee62)
        -   vllm/tool\_parsers/deepseekv4\_tool\_parser.py [deepseekv4\_tool\_parser.py](#diff-88c48d6287edf19515caf80f1506eb84b87aafd41764b6b5d7e0a7c21e172188)
    -   transformers\_utils
        -   vllm/transformers\_utils/config.py [config.py](#diff-dc4afb08b7a54bb237ee02180225d41d542166342c1f09d41978b7893f332dd0)
        -   configs
            -   vllm/transformers\_utils/configs/\_\_init\_\_.py [\_\_init\_\_.py](#diff-e8c598df4aa134a09f8b3eafb98ad77b73b8ebfbf18d589a8227bfa20ce20982)
            -   vllm/transformers\_utils/configs/deepseek\_v4.py [deepseek\_v4.py](#diff-e3f962c1e0e81ab9070cae6ae6ec7076ddc72dd90c1173df2d6db75ea4102f4f)
        -   vllm/transformers\_utils/model\_arch\_config\_convertor.py [model\_arch\_config\_convertor.py](#diff-f117da09cbc21d27e65ca0a4285b4d40116cde2f13d2bf9f23ae0348c1d01cdd)
    -   utils
        -   vllm/utils/deep\_gemm.py [deep\_gemm.py](#diff-fe4d11817ef2fcb4854bfbf8a95b1ce939d5f2f692c7a3a23793fe45e7c1ed1a)
        -   vllm/utils/flashinfer.py [flashinfer.py](#diff-3a236a5bf6563c49380c25a112f67e3fc241918dc548060b9bbfe9b4a2411e34)
        -   vllm/utils/multi\_stream\_utils.py [multi\_stream\_utils.py](#diff-eef806ffba7f1b86e4be2f9e1f5cd14d54d87162a9e61a5e411511bfa9773d4e)
    -   v1
        -   attention
            -   vllm/v1/attention/backend.py [backend.py](#diff-bdb2df4662d59d54517931b86406644ee1e0c33cb3e00afac078d2a9aa550200)
            -   backends
                -   mla
                    -   vllm/v1/attention/backends/mla/compressor\_utils.py [compressor\_utils.py](#diff-0572312d5d11ed23ea1e73abaf58242da88e1369b8412db10403bdbc0596e62f)
                    -   vllm/v1/attention/backends/mla/flashmla\_sparse.py [flashmla\_sparse.py](#diff-3b96741e32f77ce5a1ecb7212e8febaec9a3946216316dacd4bdcb3a0b26e42f)
                    -   vllm/v1/attention/backends/mla/indexer.py [indexer.py](#diff-6bcbefb4c27b923fca744f6c6e0c91c7283450a0161eaf02d82b4977507319c1)
                    -   vllm/v1/attention/backends/mla/sparse\_swa.py [sparse\_swa.py](#diff-b0dc390ae8c9dbee4d0a43c88fbea5c01f4cae4ab99c364f99c6d61815d75456)
                -   vllm/v1/attention/backends/utils.py [utils.py](#diff-64c15c2c24db107229474656b2b27d49d8d6d922ae6cd7e2d41453e7e9f796aa)
            -   ops
                -   vllm/v1/attention/ops/common.py [common.py](#diff-ab46e3226143f1a68d2590eae8b7a4836dafcb5855ecea0f54608a4efe8a9103)
                -   deepseek\_v4\_ops
                    -   vllm/v1/attention/ops/deepseek\_v4\_ops/\_\_init\_\_.py [\_\_init\_\_.py](#diff-156734a92a5b044b1ab9e11a6230de63c4acb13742cc88637b3cce3b0611b0f0)
                    -   vllm/v1/attention/ops/deepseek\_v4\_ops/cache\_utils.py [cache\_utils.py](#diff-bc8aab39edd38e34602ea06a8a5b5942ce00a2c7d0e2fc53cc0b6d544c76b5d7)
                    -   vllm/v1/attention/ops/deepseek\_v4\_ops/fused\_compress\_quant\_cache.py [fused\_compress\_quant\_cache.py](#diff-b151e83a0f27beb87532f9607fe869a9c2f4d9aff712e7469d2a13369a07beda)
                    -   vllm/v1/attention/ops/deepseek\_v4\_ops/fused\_indexer\_q.py [fused\_indexer\_q.py](#diff-1d61c4a0531cc83865351a30ed882e0a034c3ffa8cf7ebdb7bd24e9e9069f938)
                    -   vllm/v1/attention/ops/deepseek\_v4\_ops/fused\_inv\_rope\_fp8\_quant.py [fused\_inv\_rope\_fp8\_quant.py](#diff-2dec0a4af5f401994a5d66abd2af2b65eff137525446e1fb5e06105e289b2b3e)
                    -   vllm/v1/attention/ops/deepseek\_v4\_ops/fused\_qk\_rmsnorm.py [fused\_qk\_rmsnorm.py](#diff-5a750ffb54594eb0ad1734512cda86d6462ad716b452d8560795e0604acbb0e3)
        -   core
            -   vllm/v1/core/kv\_cache\_coordinator.py [kv\_cache\_coordinator.py](#diff-43875c71daa893ef7567e21633d9988c2baf95bef61e3a334a6d584d6444d725)
            -   vllm/v1/core/kv\_cache\_utils.py [kv\_cache\_utils.py](#diff-fb2b41380ca86adbba904aff40b18a0815beb73c1198818912e71723383ab604)
            -   sched
                -   vllm/v1/core/sched/interface.py [interface.py](#diff-72c31f3270a006e9ba0c9651f2b4c15d0572c616e68a78e592b426317bc0c6ba)
                -   vllm/v1/core/sched/scheduler.py [scheduler.py](#diff-9eeca590fd99f15621897e559dba39b3ec4e7c2c65ec3c3229711689e008b5f4)
            -   vllm/v1/core/single\_type\_kv\_cache\_manager.py [single\_type\_kv\_cache\_manager.py](#diff-97c184a680b7a4bd7d58b11aa0073706533cc887d990eddb98469e7374025ab3)
        -   engine
            -   vllm/v1/engine/core.py [core.py](#diff-3dd8e96bc7c1aaf28faa13b3b705f4f7bdbf755aec34dcf4d9b67c933ddfb127)
        -   vllm/v1/kv\_cache\_interface.py [kv\_cache\_interface.py](#diff-f76cdfbf02dacd9dffbfbd0d9ad68a7a6ac0d8aed70834f95a2ae8ccd2e333cb)
        -   spec\_decode
            -   vllm/v1/spec\_decode/eagle.py [eagle.py](#diff-a4809a837fbf535a8f0999b11087a53ec1c53948b50c0a1fe64396bc86de9461)
        -   worker
            -   gpu
                -   vllm/v1/worker/gpu/attn\_utils.py [attn\_utils.py](#diff-9b864c13232e1f03b906ccc83311fa78d1c37988616ae88b4afdbb5c0d186a75)
                -   vllm/v1/worker/gpu/input\_batch.py [input\_batch.py](#diff-106f39c08266f186830bb8fcd7fb1df35c0aaa5fd0ac5c17aac64aeddee48902)
                -   vllm/v1/worker/gpu/model\_runner.py [model\_runner.py](#diff-5823f988fc0264681a80db24ccaba4d364f14394815d83ec1d90944c09f571f0)
                -   model\_states
                    -   vllm/v1/worker/gpu/model\_states/default.py [default.py](#diff-aa6dbff87b81bae23ded2803a01a8c4913fceb6fe39c6d9423a20ad4b89859be)
                -   spec\_decode/eagle
                    -   vllm/v1/worker/gpu/spec\_decode/eagle/speculator.py [speculator.py](#diff-ad4e9335c0f7a88ef9f3f3e2a9f00a1546ea20b294fcb484a439fc9f1bed6720)
                    -   vllm/v1/worker/gpu/spec\_decode/eagle/utils.py [utils.py](#diff-b28be6852c553af9bcc3ebcfd33014116176c55e194640e007c28caaa6af29a4)
                -   vllm/v1/worker/gpu/states.py [states.py](#diff-31e3ae803366ca54502cafbd35fa8a9b1d25baa77a0a697791e049580e72f9c0)
            -   vllm/v1/worker/gpu\_model\_runner.py [gpu\_model\_runner.py](#diff-80ee7e2a62f9dcfbb8a312dc4e3948557e97ef187290daebbcae1e28596bda29)

## There are no files selected for viewing

[](/vllm-project/vllm/blob/main/.github/CODEOWNERS#L21)5 changes: 4 additions & 1 deletion 5 [CMakeLists.txt](#diff-1e7de1ae2d059d21e1dd75d5812d5a34b0222cef273b7c3a2af62eb747f9d20a "CMakeLists.txt")

 Show comments

[View file](/vllm-project/vllm/blob/e8e38e1686c3ca0835b9556fc1f9b28b9e1a455f/CMakeLists.txt) Edit file Delete file

[Open in desktop](https://desktop.github.com)

Original file line number

Diff line number

Diff line change

[](#diff-1e7de1ae2d059d21e1dd75d5812d5a34b0222cef273b7c3a2af62eb747f9d20a)Expand Up

@@ -291,6 +291,7 @@ set(VLLM\_EXT\_SRC

"csrc/activation\_kernels.cu"

"csrc/layernorm\_kernels.cu"

"csrc/fused\_qknorm\_rope\_kernel.cu"

"csrc/fused\_deepseek\_v4\_qnorm\_rope\_kv\_insert\_kernel.cu"

"csrc/layernorm\_quant\_kernels.cu"

"csrc/sampler.cu"

"csrc/topk.cu"

[](#diff-1e7de1ae2d059d21e1dd75d5812d5a34b0222cef273b7c3a2af62eb747f9d20a)Expand Down [](#diff-1e7de1ae2d059d21e1dd75d5812d5a34b0222cef273b7c3a2af62eb747f9d20a)Expand Up

@@ -1026,7 +1027,9 @@ endif()

set(VLLM\_MOE\_EXT\_SRC

"csrc/moe/torch\_bindings.cpp"

"csrc/moe/moe\_align\_sum\_kernels.cu"

"csrc/moe/topk\_softmax\_kernels.cu")

"csrc/moe/topk\_softmax\_kernels.cu"

"csrc/moe/topk\_softplus\_sqrt\_kernels.cu")

  

  

if(VLLM\_GPU\_LANG STREQUAL "CUDA")

list(APPEND VLLM\_MOE\_EXT\_SRC

[](#diff-1e7de1ae2d059d21e1dd75d5812d5a34b0222cef273b7c3a2af62eb747f9d20a)Expand Down

Loading

Oops, something went wrong. Retry

### Uh oh!

There was an error while loading. Please reload this page.

Toggle all file notes Toggle all file annotations

Add this suggestion to a batch that can be applied as a single commit. This suggestion is invalid because no changes were made to the code. Suggestions cannot be applied while the pull request is closed. Suggestions cannot be applied while viewing a subset of changes. Only one suggestion per line can be applied in a batch. Add this suggestion to a batch that can be applied as a single commit. Applying suggestions on deleted lines is not supported. You must change the existing code in this line in order to create a valid suggestion. Outdated suggestions cannot be applied. This suggestion has been applied or marked resolved. Suggestions cannot be applied from pending reviews. Suggestions cannot be applied on multi-line comments. Suggestions cannot be applied while the pull request is queued to merge. Suggestion cannot be applied right now. Please check back later.

## Footer

[](https://github.com)© 2026 GitHub, Inc.

### Footer navigation

-   [Terms](https://docs.github.com/site-policy/github-terms/github-terms-of-service)
-   [Privacy](https://docs.github.com/site-policy/privacy-policies/github-privacy-statement)
-   [Security](https://github.com/security)
-   [Status](https://www.githubstatus.com/)
-   [Community](https://github.community/)
-   [Docs](https://docs.github.com/)
-   [Contact](https://support.github.com?tags=dotcom-footer)
-   Manage cookies
-   Do not share my personal information

You can’t perform that action at this time.
```

---

## Assistant (Build · DeepSeek V4 Pro · 16.1s)

_Thinking:_

The upstream PR #40760 is massive - 127 Python files changed, 7 CUDA files changed, plus C++ headers, etc. This is a very large model support PR.

The GitHub pages for both the raw source and the blob view return 404, which is odd. Let me try fetching from the PR's branch or a specific commit.

From the PR files list, I can see:
- `vllm/model_executor/models/deepseek_v4.py` - the main model file
- `vllm/model_executor/layers/deepseek_v4_attention.py` - V4 attention mechanisms
- `vllm/model_executor/layers/deepseek_compressor.py` - KV compression
- `vllm/model_executor/layers/mhc.py` - Manifold-Constrained Hyper-Connections
- `vllm/v1/attention/ops/deepseek_v4_ops/` - custom attention ops
- `vllm/transformers_utils/configs/deepseek_v4.py` - config parsing
- Various kernel files and MoE backends

Let me try to fetch the deepseek_v4.py from the PR's branch/commit.

**Tool: webfetch**

**Input:**
```json
{
  "format": "markdown",
  "max_length": 60000,
  "url": "https://raw.githubusercontent.com/vllm-project/vllm/e8e38e1686c3ca0835b9556fc1f9b28b9e1a455f/vllm/model_executor/models/deepseek_v4.py"
}
```

**Output:**
```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.deepseek_v4_attention import (
    DeepseekV4Indexer,
    DeepseekV4MLAModules,
    DeepseekV4MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear, SharedFusedMoE
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import AuxStreamType
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)


class DeepseekV4FP8Config(Fp8Config):
    """FP8 config that routes MoE layers to MXFP4 quantization.

    DeepSeek V4 checkpoints use FP8 for linear/attention layers but
    MXFP4 for MoE expert weights. This config inherits standard FP8
    behavior and overrides only the MoE dispatch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_scale_e8m0: bool = True

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v4_fp8"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v4_fp8")
        ):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type == "deepseek_v4" or user_quant == "deepseek_v4_fp8":
            return "deepseek_v4_fp8"
        return None

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return Mxfp4MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)

    def is_mxfp4_quant(self, prefix, layer):
        return isinstance(layer, FusedMoE)


@triton.jit
def _deepseek_v4_stage_mega_moe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def _stage_deepseek_v4_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            "a multiple of 128."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    block_k = 128
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    _deepseek_v4_stage_mega_moe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        top_k,
        BLOCK_K=block_k,
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )


def make_deepseek_v4_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", "w1"),
            ("w2", "w2"),
            ("w3", "w3"),
        ]
    ]


class DeepseekV4MegaMoEExperts(nn.Module):
    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], object] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None

        # Register in the static forward context so the custom-op wrapper
        # can look up this module by name from within a torch.compile graph.
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _map_global_expert_id(self, expert_id: int) -> int:
        if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
            return -1
        return expert_id - self.experts_start_idx

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_id = self._map_global_expert_id(expert_id)
        if local_expert_id == -1:
            return False if return_success else None

        expert_data = param.data[local_expert_id]
        if shard_id in ("w1", "w3"):
            if "w13_" not in weight_name:
                return False if return_success else None
            shard_offset = 0 if shard_id == "w1" else self.intermediate_size
            expert_data = expert_data.narrow(0, shard_offset, self.intermediate_size)
        elif shard_id == "w2":
            if "w2_" not in weight_name:
                return False if return_success else None
        else:
            raise ValueError(f"Unsupported expert shard id: {shard_id}")

        if expert_data.shape != loaded_weight.shape:
            raise ValueError(
                f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                f"vs checkpoint {tuple(loaded_weight.shape)}"
            )
        expert_data.copy_(loaded_weight)
        return True if return_success else None

    @staticmethod
    def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
        return (sf.to(torch.int32) << 23).view(torch.float32)

    def _check_runtime_supported(self) -> None:
        if not torch.cuda.is_available():
            raise NotImplementedError("DeepSeek V4 MegaMoE requires CUDA.")
        device = self.w13_weight.device
        if device.type != "cuda":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE expert weights must be loaded on CUDA."
            )
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return

        self._check_runtime_supported()
        import vllm.third_party.deep_gemm as deep_gemm

        w13_scale = deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_uint8_to_float(self.w13_weight_scale.data).contiguous(),
            2 * self.intermediate_size,
            self.hidden_size,
            (1, 32),
            self.num_local_experts,
        )
        w2_scale = deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_uint8_to_float(self.w2_weight_scale.data).contiguous(),
            self.hidden_size,
            self.intermediate_size,
            (1, 32),
            self.num_local_experts,
        )
        self._transformed_l1_weights, self._transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
                (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
            )
        )
        # Drop the original loader-side parameters: the MegaMoE kernels only
        # consume the transformed views above. transform_weights_for_mega_moe
        # allocates a fresh tensor for the L1 weight (see _interleave_l1_weights)
        # and fresh SF tensors for L1/L2; the L2 weight is the only tensor that
        # aliases the original storage, and _transformed_l2_weights still holds
        # it, so the storage stays live after we drop the Parameter.
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None

    def get_symm_buffer(self):
        import vllm.third_party.deep_gemm as deep_gemm

        group = get_ep_group().device_group
        device = torch.cuda.current_device()
        key = (
            id(group),
            device,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but the symmetric buffer was sized for {self.max_num_tokens}."
            )
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        torch.ops.vllm.deepseek_v4_mega_moe_experts(
            hidden_states,
            topk_weights,
            topk_ids,
            y,
            self.prefix,
            activation_clamp,
            fast_math,
        )
        return y

    def _run_mega_moe(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        y: torch.Tensor,
        activation_clamp: float | None,
        fast_math: bool,
    ) -> None:
        import vllm.third_party.deep_gemm as deep_gemm

        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        _stage_deepseek_v4_mega_moe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
        )

        # This method must have been already called duing the weight loading phase.
        # We call it again here to cover the dummy weight loading case.
        self.finalize_weights()

        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        deep_gemm.fp8_fp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )


DeepseekV4MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def _deepseek_v4_mega_moe_experts_op(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
    activation_clamp: float | None,
    fast_math: bool,
) -> None:
    self = get_forward_context().no_compile_layers[layer_name]
    self._run_mega_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        out,
        activation_clamp,
        fast_math,
    )


def _deepseek_v4_mega_moe_experts_op_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
    activation_clamp: float | None,
    fast_math: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_mega_moe_experts",
    op_func=_deepseek_v4_mega_moe_experts_op,
    mutates_args=["out"],
    fake_impl=_deepseek_v4_mega_moe_experts_op_fake,
)


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix
        if vllm_config.parallel_config.enable_expert_parallel:
            self.use_mega_moe = envs.VLLM_DEEPSEEK_V4_USE_MEGA_MOE
        else:
            self.use_mega_moe = False

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.use_mega_moe and self.scoring_func != "sqrtsoftplus":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently supports sqrtsoftplus routing only."
            )

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32

        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=self.use_mega_moe,
                prefix=f"{prefix}.shared_experts",
            )

        if self.use_mega_moe:
            self._init_mega_moe_experts(vllm_config, config, prefix)
        else:
            self._init_shared_fused_moe_experts(config, quant_config, prefix)

    def _init_mega_moe_experts(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str,
    ) -> None:
        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.world_size
        self.ep_rank = self.ep_group.rank_in_group
        assert config.n_routed_experts % self.ep_size == 0

        self.n_local_experts = config.n_routed_experts // self.ep_size
        self.experts_start_idx = self.ep_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.experts = DeepseekV4MegaMoEExperts(
            vllm_config,
            num_experts=config.n_routed_experts,
            num_local_experts=self.n_local_experts,
            experts_start_idx=self.experts_start_idx,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
        )

    def _init_shared_fused_moe_experts(
        self,
        config,
        quant_config,
        prefix: str,
    ) -> None:
        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0

        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None:
            if input_ids is None:
                raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
            input_ids = input_ids.to(dtype=self.hash_indices_dtype)
        if not self.use_mega_moe:
            return self._forward_shared_fused_moe(hidden_states, input_ids)

        org_shape = hidden_states.shape
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        activation_clamp = (
            float(self.swiglu_limit) if self.swiglu_limit is not None else None
        )
        final_hidden_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
            activation_clamp=activation_clamp,
        )

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            final_hidden_states += shared_output

        return final_hidden_states.view(org_shape)

    def _forward_shared_fused_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        if self.experts.is_internal_router:
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states += shared_output

        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(org_shape)

    def finalize_mega_moe_weights(self) -> None:
        if self.use_mega_moe:
            self.experts.finalize_weights()


class DeepseekV4Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.n_heads % tp_size == 0

        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # NOTE(zyongye) Compress ratio can't be 0
        # we do this for because MTP layer is not included
        # in the compress ratio list
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings

        # Padded to min 64 heads for FlashMLA, initialized to -inf
        # (no sink effect). Weight loading fills the first n_local_heads slots.
        padded_heads = max(self.n_local_heads, 64)
        self.attn_sink = nn.Parameter(
            torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = config.quantization_config["scale_fmt"]

        self.rope_parameters = config.rope_scaling

        # Initialize rotary embedding BEFORE DeepseekV4MLAModules (which needs it)
        rope_parameters = config.rope_parameters
        rope_parameters["rope_theta"] = (
            config.compress_rope_theta if self.compress_ratio > 1 else config.rope_theta
        )
        # TODO(yifan): double check this!
        # rope_parameters["rope_type"] = "deepseek_yarn"
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        rope_parameters["mscale"] = 0  # Disable mscale
        rope_parameters["mscale_all_dim"] = 0  # Disable mscale
        rope_parameters["is_deepseek_v4"] = True
        rope_parameters["rope_dim"] = self.rope_head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
            dtype=config.torch_dtype,
        )

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=vllm_config.cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
            )

        mla_modules = DeepseekV4MLAModules(
            vllm_config=vllm_config,
            fused_wqa_wkv=self.fused_wqa_wkv,
            q_norm=self.q_norm,
            wq_b=self.wq_b,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            rotary_emb=self.rotary_emb,
            indexer=self.indexer,
            indexer_rotary_emb=self.rotary_emb,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream=aux_stream,
        )
        self.mla_attn = DeepseekV4MultiHeadLatentAttentionWrapper(
            hidden_size=self.hidden_size,
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.softmax_scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            v_head_dim=self.head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.head_dim,
            o_lora_rank=self.o_lora_rank,
            mla_modules=mla_modules,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ):
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4Attention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream=aux_stream_dict.get(AuxStreamType.Attention)
            if aux_stream_dict is not None
            else None,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        # Lazy import to avoid top-level tilelang dependency.
        # Registers both torch.ops.vllm.mhc_pre and mhc_post,
        # so hc_post() doesn't need its own import.
        import vllm.model_executor.layers.mhc  # noqa: F401

        post_mix, res_mix, layer_input = torch.ops.vllm.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return torch.ops.vllm.mhc_post(x, residual, post, comb)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


@support_torch_compile
class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(1)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
        }

        self.device = current_platform.device_type
        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_dict=self.aux_stream_dict,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)

        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(
                self.hc_mult,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

        # Pre-hc_head residual stream buffer for the MTP draft. Stable
        # address (outside the cudagraph pool) so the copy_ in forward()
        # refreshes it correctly across captured shapes.
        self._mtp_hidden_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.hc_dim,
            dtype=vllm_config.model_config.dtype,
            device=self.device,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(
                hidden_states,
                positions,
                input_ids,
            )

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)

    def finalize_mega_moe_weights(self) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.ffn.finalize_mega_moe_weights()


@torch.compile(backend=current_platform.simple_compile_backend)
def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    x = hidden_states
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


class DeepseekV4ForCausalLM(nn.Module):
    model_cls = DeepseekV4Model

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex={
            # Routed MoE expert scales: experts.N.wX.scale -> .weight_scale
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # Everything else (FP8 linear + shared experts): .scale -> .weight_scale_inv
            re.compile(r"\.scale$"): ".weight_scale_inv",
        },
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".attn.compressor.": ".attn.mla_attn.compressor.",
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

```

**Tool: webfetch**

**Input:**
```json
{
  "format": "markdown",
  "max_length": 30000,
  "url": "https://raw.githubusercontent.com/vllm-project/vllm/e8e38e1686c3ca0835b9556fc1f9b28b9e1a455f/vllm/model_executor/layers/quantization/__init__.py"
}
```

**Output:**
```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "gguf",
    "gptq_marlin",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "mxfp8",
    "cpu_awq",
    "online",
    # Below are values of the OnlineQuantScheme enum, specified as strings to
    # avoid circular import issues. This is here to provide a shortcut where
    # the user can specify "LLM(..., quantization='fp8_per_tensor')" as
    # shorthand for creating a more complicated online quant config object
    "fp8_per_tensor",
    "fp8_per_block",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
    "tpu_int8",
    "fbgemm_fp8",
    "fp_quant",
    "experts_int8",
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    from vllm.config.quantization import OnlineQuantScheme
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4FP8Config

    from .awq import AWQConfig
    from .awq_marlin import AWQMarlinConfig
    from .bitsandbytes import BitsAndBytesConfig
    from .compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from .cpu_wna16 import CPUAWQConfig
    from .experts_int8 import ExpertsInt8Config
    from .fbgemm_fp8 import FBGEMMFp8Config
    from .fp8 import Fp8Config
    from .fp_quant import FPQuantConfig
    from .gguf import GGUFConfig
    from .gptq import GPTQConfig
    from .gptq_marlin import GPTQMarlinConfig
    from .inc import INCConfig
    from .modelopt import (
        ModelOptFp8Config,
        ModelOptMixedPrecisionConfig,
        ModelOptMxFp8Config,
        ModelOptNvFp4Config,
    )
    from .moe_wna16 import MoeWNA16Config
    from .mxfp4 import GptOssMxfp4Config, Mxfp4Config
    from .mxfp8 import Mxfp8Config
    from .online.base import OnlineQuantizationConfig
    from .torchao import TorchAOConfig

    method_to_config: dict[str, type[QuantizationConfig]] = {
        "awq": AWQConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "fp_quant": FPQuantConfig,
        "modelopt": ModelOptFp8Config,
        "modelopt_fp4": ModelOptNvFp4Config,
        "modelopt_mxfp8": ModelOptMxFp8Config,
        "modelopt_mixed": ModelOptMixedPrecisionConfig,
        "gguf": GGUFConfig,
        "gptq_marlin": GPTQMarlinConfig,
        "awq_marlin": AWQMarlinConfig,
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "experts_int8": ExpertsInt8Config,
        "quark": QuarkConfig,
        "moe_wna16": MoeWNA16Config,
        "torchao": TorchAOConfig,
        "auto-round": INCConfig,
        "inc": INCConfig,
        "mxfp4": Mxfp4Config,
        "gpt_oss_mxfp4": GptOssMxfp4Config,
        "deepseek_v4_fp8": DeepseekV4FP8Config,
        "mxfp8": Mxfp8Config,
        "cpu_awq": CPUAWQConfig,
        "online": OnlineQuantizationConfig,
    }

    # Below are values of the OnlineQuantScheme enum. This is here to provide
    # a shortcut where the user can specify
    # "LLM(..., quantization='fp8_per_tensor')" as shorthand for creating a
    # more complicated online quant config object
    for scheme in OnlineQuantScheme:
        assert scheme.value not in method_to_config, (
            f"Online quant scheme {scheme.value!r} conflicts with an "
            f"existing quantization method"
        )
        method_to_config[scheme.value] = OnlineQuantizationConfig

    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]

```

---

## Assistant (Build · DeepSeek V4 Pro · 227.9s)

_Thinking:_

Now I have the complete picture of how upstream vLLM handles DeepSeek V4. Let me analyze the key findings:

## Key Findings from Upstream Implementation

### 1. Custom Quantization Config: `DeepseekV4FP8Config`

This is the critical piece that solves the mxfp4 problem! It's a subclass of `Fp8Config` that:

- `get_name()` returns `"deepseek_v4_fp8"` - a new quantization method name
- `override_quantization_method()`:
  - Detects config.json's `quant_method: "fp8"` 
  - Checks if `model_type == "deepseek_v4"` from the HF config
  - Returns `"deepseek_v4_fp8"` to take over
- `get_quant_method()`:
  - For `FusedMoE` layers: returns `Mxfp4MoEMethod` (routes MoE to MXFP4!)
  - For other layers: calls `super().get_quant_method()` (standard FP8)
- `is_mxfp4_quant()` returns True for FusedMoE layers

This elegantly solves the mixed FP8+FP4 problem: FP8 for attention/dense, MXFP4 for MoE experts.

### 2. Model Registration

In the upstream `registry.py`, there must be an entry mapping:
- `"DeepseekV4ForCausalLM"` → `("deepseek_v4", "DeepseekV4ForCausalLM")`

### 3. MegaMoE Experts

The V4 model has a special `DeepseekV4MegaMoEExperts` class with:
- Direct MXFP4 weight parameters (uint8 packed weights + uint8 E8M0 scales)
- Uses DeepGEMM for execution (`deep_gemm.fp8_fp4_mega_moe`)
- Block size 32 (hidden_size // 32 for scales)
- Requires SM100 (Blackwell) GPUs

### 4. New Architecture Components
- `DeepseekV4MultiHeadLatentAttentionWrapper` - new attention
- `DeepseekV4MLAModules` - MLA with indexer/compressor support
- `DeepseekV4Indexer` - sparse attention indexer
- mHC (Manifold-Constrained Hyper-Connections) via `mhc.py`
- Compressor layers for c4a/c128a attention
- Custom CUDA kernels for fused operations

### 5. Weight Loading
The `hf_to_vllm_mapper` handles important remappings:
- Expert weight scales: `experts.N.wX.scale` → `experts.N.wX.weight_scale`
- Regular scales: `.scale` → `.weight_scale_inv`
- E8M0 float8→uint8 conversion for MoE scales

Now let me provide a comprehensive summary to the user about what needs to be done to add DeepSeek V4 support to this Lvllm fork, focusing on the MXFP4 quantization aspect.

The good news is that the quantization problem is elegantly solved by `DeepseekV4FP8Config` - a config that inherits FP8 for dense layers but routes MoE to MXFP4. The bad news is that the full model implementation is very large and interconnected.

For the user's 16GB VRAM target:
- DeepSeek-V4-Flash uses 74 GiB memory on GPU (from the log: "Model loading took 74.05 GiB memory")
- This is way beyond 16GB VRAM
- They'd need CPU offloading, which the NUMA interleaving can help with
- The MegaMoE path requires SM100 which RTX PRO 2000 may or may not have

Let me check: RTX PRO 2000 is a Blackwell (RTX 50 series) GPU. Blackwell is SM120 (Blackwell), not SM100 (which is B200/B300). Wait, actually Blackwell is compute capability 12.x = SM120. RTX PRO 2000 is a professional Blackwell GPU.

But the code says: `if torch.cuda.get_device_capability(device)[0] != 10: raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")` - this checks for SM100, not SM120.

So there's an architecture compatibility issue. The MegaMoE path won't work on SM120. The model would need the SharedFusedMoE fallback path instead.

Let me give the user a comprehensive, actionable analysis now.

Now I have the complete picture. Here's a comprehensive analysis:

---

## How Upstream vLLM Solved the MXFP4 Problem

### The Key: `DeepseekV4FP8Config`

Upstream created a **custom quantization config** (`deepseek_v4.py:58-103`) that inherits from `Fp8Config` but **routes MoE layers to MXFP4**:

```python
class DeepseekV4FP8Config(Fp8Config):
    def get_name(self):
        return "deepseek_v4_fp8"  # new quantization method name

    def override_quantization_method(self, hf_quant_cfg, user_quant, hf_config):
        # Auto-detect: checks config.json's quant_method=="fp8" AND model_type=="deepseek_v4"
        if hf_config.model_type == "deepseek_v4":
            return "deepseek_v4_fp8"

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):       # ← THIS IS THE KEY LINE
            return Mxfp4MoEMethod(...)         # MoE → MXFP4
        return super().get_quant_method(...)   # Everything else → FP8
```

This means: attention layers and dense MLPs use standard FP8 quantization, but MoE expert weights automatically use the existing `Mxfp4MoEMethod` (which supports MARLIN, FlashInfer TRTLLM, etc. backends).

### Weight Format

V4 stores MoE weights as **packed uint8** (2 FP4 values per byte) with **uint8 E8M0 scales** (block_size=32). The weight loader converts `float8_e8m0fnu` scale tensors to raw `uint8`:

```python
if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
    loaded_weight = loaded_weight.view(torch.uint8)
```

---

## What Needs to Happen in This Lvllm Fork

There are **three tiers** of work, ranked by complexity:

### Tier 1: Quantization Config Only (Minimal — ~50 lines)

You can add `DeepseekV4FP8Config` **right now** without the full model. This lets the quantization auto-detection work:

1. Add `"deepseek_v4_fp8"` to `QuantizationMethods` in `__init__.py`
2. Create the `DeepseekV4FP8Config` class (subclass `Fp8Config`, override `get_quant_method`)
3. Register it in `_verify_quantization()` overrides list

However, this only solves quant detection — the model architecture still won't load.

### Tier 2: Port Upstream Model (~100+ files, 5000+ lines)

The PR #40760 touches 127 Python files, 7 CUDA files. This is a major port involving:
- `deepseek_v4.py` — main model (~850 lines)
- `deepseek_v4_attention.py` — attention with compressor/indexer
- `deepseek_compressor.py` — KV compression (c4a/c128a)
- `mhc.py` — Manifold-Constrained Hyper-Connections
- `deepseek_v4_ops/` — fused attention kernels
- Custom CUDA kernels + DeepGEMM/MegaMoE
- Config changes, registry, tokenizer, tool parser

### Tier 3: Memory Optimization for 16GB VRAM

Upstream logs show V4-Flash takes **74 GiB GPU memory**. For 16GB VRAM you need:
- `--cpu-offload-gb` with NUMA interleaving (your existing plugin)
- Expert offloading (`--enable-expert-parallel` with CPU fallback)
- Small `--max-model-len` (not 1M)
- Possibly `--gpu-memory-utilization` tuned low

---

## Concrete Plan

**Tool: question**

**Input:**
```json
{
  "questions": [
    {
      "question": "The upstream DeepSeek V4 implementation is a large PR (127 files). For your Lvllm fork, which approach should I take?",
      "header": "Implementation strategy",
      "options": [
        {
          "label": "Start with quant config only",
          "description": "First add DeepseekV4FP8Config and quantization auto-detection. Quick win that unblocks quant loading."
        },
        {
          "label": "Port the full upstream model",
          "description": "Port deepseek_v4.py, attention, compressor, mhc, kernels, and all dependencies. Multi-day effort."
        },
        {
          "label": "Both — quant first, then model",
          "description": "Start with the quantization config (1-2 files), then progressively port model architecture."
        }
      ]
    }
  ]
}
```

**Output:**
```
User has answered your questions: "The upstream DeepSeek V4 implementation is a large PR (127 files). For your Lvllm fork, which approach should I take?"="Both — quant first, then model". You can now continue with the user's answers in mind.
```

---

## Assistant (Build · DeepSeek V4 Pro · 14.2s)

_Thinking:_

The user wants both - start with the quantization config first, then progressively port the model architecture.

Let me plan this out:

### Step 1: Quantization Config
Add `DeepseekV4FP8Config` to this codebase. This requires:
1. Add `"deepseek_v4_fp8"` to the `QuantizationMethods` Literal in `__init__.py`
2. Add `DeepseekV4FP8Config` class - either in a new file or inline
3. Register it in the `get_quantization_config()` function
4. Add it to the overrides list in `config/model.py` `_verify_quantization()`

But wait - looking at the upstream, DeepseekV4FP8Config is in `deepseek_v4.py` and imported in `get_quantization_config()`. For the local fork, I should either:
- Create a standalone `deepseek_v4_fp8.py` quantization config
- Or add it to the existing `deepseek_v4.py` model file (once we create it)

For the quant-first approach, I'll put it in a new file so it doesn't depend on the model code.

Let me also check what the local `__init__.py` looks like for the quantization methods to see what needs modifying.

Actually, let me read the relevant files first to understand exactly what changes are needed.

Let me read the key files I need to modify for the quantization config.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/__init__.py"
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/__init__.py</path>
<type>file</type>
<content>
1: # SPDX-License-Identifier: Apache-2.0
2: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
3: 
4: from typing import Literal, get_args
5: 
6: from vllm.logger import init_logger
7: from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
8: from vllm.platforms import current_platform
9: 
10: logger = init_logger(__name__)
11: 
12: QuantizationMethods = Literal[
13:     "awq",
14:     "fp8",
15:     "fbgemm_fp8",
16:     "fp_quant",
17:     "modelopt",
18:     "modelopt_fp4",
19:     "modelopt_mxfp8",
20:     "modelopt_mixed",
21:     "gguf",
22:     "gptq_marlin",
23:     "awq_marlin",
24:     "gptq",
25:     "compressed-tensors",
26:     "bitsandbytes",
27:     "experts_int8",
28:     "quark",
29:     "moe_wna16",
30:     "torchao",
31:     "inc",
32:     "mxfp4",
33:     "mxfp8",
34:     "cpu_awq",
35:     "online",
36:     # Below are values of the OnlineQuantScheme enum, specified as strings to
37:     # avoid circular import issues. This is here to provide a shortcut where
38:     # the user can specify "LLM(..., quantization='fp8_per_tensor')" as
39:     # shorthand for creating a more complicated online quant config object
40:     "fp8_per_tensor",
41:     "fp8_per_block",
42: ]
43: QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))
44: 
45: DEPRECATED_QUANTIZATION_METHODS = [
46:     "tpu_int8",
47:     "fbgemm_fp8",
48:     "fp_quant",
49:     "experts_int8",
50: ]
51: 
52: # The customized quantization methods which will be added to this dict.
53: _CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}
54: 
55: 
56: def register_quantization_config(quantization: str):
57:     """Register a customized vllm quantization config.
58: 
59:     When a quantization method is not supported by vllm, you can register a customized
60:     quantization config to support it.
61: 
62:     Args:
63:         quantization (str): The quantization method name.
64: 
65:     Examples:
66:         >>> from vllm.model_executor.layers.quantization import (
67:         ...     register_quantization_config,
68:         ... )
69:         >>> from vllm.model_executor.layers.quantization import get_quantization_config
70:         >>> from vllm.model_executor.layers.quantization.base_config import (
71:         ...     QuantizationConfig,
72:         ... )
73:         >>>
74:         >>> @register_quantization_config("my_quant")
75:         ... class MyQuantConfig(QuantizationConfig):
76:         ...     pass
77:         >>>
78:         >>> get_quantization_config("my_quant")
79:         <class 'MyQuantConfig'>
80:     """  # noqa: E501
81: 
82:     def _wrapper(quant_config_cls):
83:         if quantization in QUANTIZATION_METHODS:
84:             logger.warning(
85:                 "The quantization method '%s' already exists and will be "
86:                 "overwritten by the quantization config %s.",
87:                 quantization,
88:                 quant_config_cls,
89:             )
90:         else:
91:             QUANTIZATION_METHODS.append(quantization)
92:             # Automatically assume the custom quantization config is supported
93:             if sq := current_platform.supported_quantization:
94:                 sq.append(quantization)
95: 
96:         if not issubclass(quant_config_cls, QuantizationConfig):
97:             raise ValueError(
98:                 "The quantization config must be a subclass of `QuantizationConfig`."
99:             )
100:         _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
101:         return quant_config_cls
102: 
103:     return _wrapper
104: 
105: 
106: def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
107:     if quantization not in QUANTIZATION_METHODS:
108:         raise ValueError(f"Invalid quantization method: {quantization}")
109: 
110:     # lazy import to avoid triggering `torch.compile` too early
111:     from vllm.config.quantization import OnlineQuantScheme
112:     from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
113: 
114:     from .awq import AWQConfig
115:     from .awq_marlin import AWQMarlinConfig
116:     from .bitsandbytes import BitsAndBytesConfig
117:     from .compressed_tensors.compressed_tensors import (
118:         CompressedTensorsConfig,
119:     )
120:     from .cpu_wna16 import CPUAWQConfig
121:     from .experts_int8 import ExpertsInt8Config
122:     from .fbgemm_fp8 import FBGEMMFp8Config
123:     from .fp8 import Fp8Config
124:     from .fp_quant import FPQuantConfig
125:     from .gguf import GGUFConfig
126:     from .gptq import GPTQConfig
127:     from .gptq_marlin import GPTQMarlinConfig
128:     from .inc import INCConfig
129:     from .modelopt import (
130:         ModelOptFp8Config,
131:         ModelOptMixedPrecisionConfig,
132:         ModelOptMxFp8Config,
133:         ModelOptNvFp4Config,
134:     )
135:     from .moe_wna16 import MoeWNA16Config
136:     from .mxfp4 import Mxfp4Config
137:     from .mxfp8 import Mxfp8Config
138:     from .online.base import OnlineQuantizationConfig
139:     from .torchao import TorchAOConfig
140: 
141:     method_to_config: dict[str, type[QuantizationConfig]] = {
142:         "awq": AWQConfig,
143:         "fp8": Fp8Config,
144:         "fbgemm_fp8": FBGEMMFp8Config,
145:         "fp_quant": FPQuantConfig,
146:         "modelopt": ModelOptFp8Config,
147:         "modelopt_fp4": ModelOptNvFp4Config,
148:         "modelopt_mxfp8": ModelOptMxFp8Config,
149:         "modelopt_mixed": ModelOptMixedPrecisionConfig,
150:         "gguf": GGUFConfig,
151:         "gptq_marlin": GPTQMarlinConfig,
152:         "awq_marlin": AWQMarlinConfig,
153:         "gptq": GPTQConfig,
154:         "compressed-tensors": CompressedTensorsConfig,
155:         "bitsandbytes": BitsAndBytesConfig,
156:         "experts_int8": ExpertsInt8Config,
157:         "quark": QuarkConfig,
158:         "moe_wna16": MoeWNA16Config,
159:         "torchao": TorchAOConfig,
160:         "auto-round": INCConfig,
161:         "inc": INCConfig,
162:         "mxfp4": Mxfp4Config,
163:         "mxfp8": Mxfp8Config,
164:         "cpu_awq": CPUAWQConfig,
165:         "online": OnlineQuantizationConfig,
166:     }
167: 
168:     # Below are values of the OnlineQuantScheme enum. This is here to provide
169:     # a shortcut where the user can specify
170:     # "LLM(..., quantization='fp8_per_tensor')" as shorthand for creating a
171:     # more complicated online quant config object
172:     for scheme in OnlineQuantScheme:
173:         assert scheme.value not in method_to_config, (
174:             f"Online quant scheme {scheme.value!r} conflicts with an "
175:             f"existing quantization method"
176:         )
177:         method_to_config[scheme.value] = OnlineQuantizationConfig
178: 
179:     # Update the `method_to_config` with customized quantization methods.
180:     method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)
181: 
182:     return method_to_config[quantization]
183: 
184: 
185: __all__ = [
186:     "QuantizationConfig",
187:     "QuantizationMethods",
188:     "get_quantization_config",
189:     "register_quantization_config",
190:     "QUANTIZATION_METHODS",
191: ]

(End of file - total 191 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/config/model.py",
  "offset": 920,
  "limit": 110
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/config/model.py</path>
<type>file</type>
<content>
920:         if convert_type != "none":
921:             logger.info(
922:                 "Resolved `--convert auto` to `--convert %s`. "
923:                 "Pass the value explicitly to silence this message.",
924:                 convert_type,
925:             )
926: 
927:         return convert_type
928: 
929:     def _verify_quantization(self) -> None:
930:         supported_quantization = me_quant.QUANTIZATION_METHODS
931:         if self.quantization is not None:
932:             self.quantization = cast(me_quant.QuantizationMethods, self.quantization)
933: 
934:         # Parse quantization method from the HF model config, if available.
935:         quant_cfg = self.model_arch_config.quantization_config
936: 
937:         if quant_cfg is not None:
938:             quant_method = quant_cfg["quant_method"]
939:             # Quantization methods which are overrides (i.e. they have a
940:             # `override_quantization_method` method) must be checked in order
941:             # of preference (this is particularly important for GPTQ).
942:             overrides = [
943:                 "gptq_marlin",
944:                 "awq_marlin",
945:                 "inc",
946:                 "moe_wna16",
947:                 "modelopt",
948:                 "modelopt_fp4",
949:                 "modelopt_mxfp8",
950:                 "modelopt_mixed",
951:                 # Ensure heavy backends are probed last to avoid unnecessary
952:                 # imports during override detection (e.g., MXFP4 imports Triton)
953:                 "mxfp4",
954:                 "cpu_awq",
955:                 "gguf",
956:             ]
957:             quantization_methods = [
958:                 q for q in supported_quantization if q not in overrides
959:             ]
960:             # Any custom overrides will be in quantization_methods so we place
961:             # them at the start of the list so custom overrides have preference
962:             # over the built-in ones.
963:             quantization_methods = quantization_methods + overrides
964: 
965:             # Detect which checkpoint is it
966:             for name in quantization_methods:
967:                 method = me_quant.get_quantization_config(name)
968:                 quantization_override = method.override_quantization_method(
969:                     quant_cfg, self.quantization
970:                 )
971:                 if quantization_override is not None:
972:                     # Raise error if the override is not custom (custom would
973:                     # be in QUANTIZATION_METHODS but not QuantizationMethods)
974:                     # and hasn't been added to the overrides list.
975:                     if (
976:                         name in get_args(me_quant.QuantizationMethods)
977:                         and name not in overrides
978:                     ):
979:                         raise ValueError(
980:                             f"Quantization method {name} is an override but "
981:                             "is has not been added to the `overrides` list "
982:                             "above. This is necessary to ensure that the "
983:                             "overrides are checked in order of preference."
984:                         )
985:                     quant_method = quantization_override
986:                     self.quantization = quantization_override
987:                     break
988: 
989:             quant_method = quant_method if quant_method != "" else None
990:             # Verify quantization configurations.
991:             if self.quantization is None:
992:                 self.quantization = quant_method
993:             elif self.quantization != quant_method:
994:                 raise ValueError(
995:                     "Quantization method specified in the model config "
996:                     f"({quant_method}) does not match the quantization "
997:                     f"method specified in the `quantization` argument "
998:                     f"({self.quantization})."
999:                 )
1000: 
1001:         if self.quantization is not None:
1002:             if self.quantization not in supported_quantization:
1003:                 raise ValueError(
1004:                     f"Unknown quantization method: {self.quantization}. Must "
1005:                     f"be one of {supported_quantization}."
1006:                 )
1007:             current_platform.verify_quantization(self.quantization)
1008: 
1009:         if self.quantization in me_quant.DEPRECATED_QUANTIZATION_METHODS:
1010:             if self.allow_deprecated_quantization:
1011:                 logger.warning(
1012:                     "The quantization method %s is deprecated "
1013:                     "and will be removed in future versions of vLLM.",
1014:                     self.quantization,
1015:                 )
1016:             else:
1017:                 raise ValueError(
1018:                     "The quantization method %s is deprecated "
1019:                     "and will be removed in future versions of vLLM. To bypass, "
1020:                     "set `--allow-deprecated-quantization`.",
1021:                     self.quantization,
1022:                 )
1023: 
1024:     def _verify_cuda_graph(self) -> None:
1025:         # CUDAGraph capture not supported for encoder-decoder models on ROCm
1026:         unsupported_rocm = self.is_encoder_decoder
1027:         if unsupported_rocm and not self.enforce_eager and current_platform.is_rocm():
1028:             logger.warning(
1029:                 "CUDA graph is not supported for %s on ROCm yet, fallback "

(Showing lines 920-1029 of 2188. Use offset=1030 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/mxfp4.py",
  "offset": 1,
  "limit": 50
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/mxfp4.py</path>
<type>file</type>
<content>
1: # SPDX-License-Identifier: Apache-2.0
2: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
3: 
4: import torch
5: 
6: from vllm.config import get_current_vllm_config
7: from vllm.logger import init_logger
8: from vllm.model_executor.layers.attention import Attention
9: from vllm.model_executor.layers.fused_moe import (
10:     FusedMoE,
11:     FusedMoEConfig,
12:     FusedMoEMethodBase,
13: )
14: from vllm.model_executor.layers.fused_moe import modular_kernel as mk
15: from vllm.model_executor.layers.fused_moe.config import (
16:     FusedMoEParallelConfig,
17:     FusedMoEQuantConfig,
18: )
19: from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
20:     TRITON_BACKENDS,
21:     Mxfp4MoeBackend,
22:     convert_to_mxfp4_moe_kernel_format,
23:     make_mxfp4_moe_kernel,
24:     make_mxfp4_moe_quant_config,
25:     mxfp4_round_up_hidden_size_and_intermediate_size,
26:     select_mxfp4_moe_backend,
27: )
28: from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
29: from vllm.model_executor.layers.quantization import QuantizationMethods
30: from vllm.model_executor.layers.quantization.base_config import (
31:     QuantizationConfig,
32:     QuantizeMethodBase,
33: )
34: from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
35: from vllm.model_executor.utils import replace_parameter, set_weight_attrs
36: 
37: logger = init_logger(__name__)
38: 
39: 
40: class Mxfp4Config(QuantizationConfig):
41:     def __init__(self, ignored_layers: list[str] | None = None):
42:         super().__init__()
43:         self.ignored_layers = ignored_layers
44: 
45:     @classmethod
46:     def from_config(cls, config):
47:         return cls()
48: 
49:     @classmethod
50:     def get_min_capability(cls) -> int:

(Showing lines 1-50 of 413. Use offset=51 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4 Pro · 18.8s)

_Thinking:_

Now I have all the information I need to implement the quantization config. Let me also check the registry.py to understand what model entries exist, and the `Mxfp4MoEMethod` class to understand its interface.

Let me also read the remaining parts of mxfp4.py to understand Mxfp4MoEMethod's constructor.

Actually, looking at what I need to implement:

### Step 1: Add `deepseek_v4_fp8` quantization method

1. **`__init__.py`** (quantization):
   - Add `"deepseek_v4_fp8"` to the `QuantizationMethods` Literal
   - Add `DeepseekV4FP8Config` import and entry in `get_quantization_config()`

2. **Create `deepseek_v4_fp8.py`** quantization config file:
   - `DeepseekV4FP8Config(Fp8Config)` that overrides `get_quant_method()` to route FusedMoE to Mxfp4MoEMethod

3. **`config/model.py`**:
   - Add `"deepseek_v4_fp8"` to the `overrides` list in `_verify_quantization()`

But wait - I need to check if `DeepseekV4FP8Config.override_quantization_method()` also needs access to `hf_config.model_type`. Let me check the current signature of `override_quantization_method`.

Let me read the base class to see the method signature.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/mxfp4.py",
  "offset": 50,
  "limit": 200
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/mxfp4.py</path>
<type>file</type>
<content>
50:     def get_min_capability(cls) -> int:
51:         return 80
52: 
53:     @classmethod
54:     def get_name(cls) -> QuantizationMethods:
55:         return "mxfp4"
56: 
57:     @classmethod
58:     def get_supported_act_dtypes(cls) -> list[torch.dtype]:
59:         return [torch.bfloat16]
60: 
61:     @classmethod
62:     def get_config_filenames(cls) -> list[str]:
63:         return []
64: 
65:     def get_quant_method(
66:         self, layer: torch.nn.Module, prefix: str
67:     ) -> "QuantizeMethodBase | None":
68:         if isinstance(layer, LinearBase):
69:             if self.ignored_layers and is_layer_skipped(
70:                 prefix=prefix,
71:                 ignored_layers=self.ignored_layers,
72:                 fused_mapping=self.packed_modules_mapping,
73:             ):
74:                 return UnquantizedLinearMethod()
75:             logger.debug_once(
76:                 "MXFP4 linear layer is not implemented - falling back to "
77:                 "UnquantizedLinearMethod.",
78:                 scope="local",
79:             )
80:             return UnquantizedLinearMethod()
81:         elif isinstance(layer, FusedMoE):
82:             return Mxfp4MoEMethod(layer.moe_config)
83:         elif isinstance(layer, Attention):
84:             logger.debug_once(
85:                 "MXFP4 attention layer is not implemented. "
86:                 "Skipping quantization for this layer.",
87:                 scope="local",
88:             )
89:         return None
90: 
91:     def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
92:         """MXFP4 config always uses MXFP4 quantization."""
93:         return True
94: 
95: 
96: class Mxfp4MoEMethod(FusedMoEMethodBase):
97:     """MXFP4 MoE quantization method."""
98: 
99:     def __init__(self, moe: FusedMoEConfig):
100:         super().__init__(moe)
101:         self.weight_dtype = "mxfp4"
102:         self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)
103: 
104:         self.max_capture_size = (
105:             get_current_vllm_config().compilation_config.max_cudagraph_capture_size
106:         )
107: 
108:         self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
109:         self.moe_kernel: mk.FusedMoEKernel | None = None
110: 
111:         # Used for triton kernel precision configs
112:         self.w13_precision_config = None
113:         self.w2_precision_config = None
114: 
115:     @property
116:     def skip_forward_padding(self) -> bool:
117:         # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
118:         # so can skip the padding in the forward before applying the moe method
119:         return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8
120: 
121:     def maybe_roundup_sizes(
122:         self,
123:         hidden_size: int,
124:         intermediate_size_per_partition: int,
125:         act_dtype: torch.dtype,
126:         moe_parallel_config: FusedMoEParallelConfig,
127:     ) -> tuple[int, int]:
128:         hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
129:             hidden_size=hidden_size,
130:             intermediate_size_per_partition=intermediate_size_per_partition,
131:             act_dtype=act_dtype,
132:             moe_parallel_config=moe_parallel_config,
133:         )
134:         return mxfp4_round_up_hidden_size_and_intermediate_size(
135:             self.mxfp4_backend, hidden_size, intermediate_size_per_partition
136:         )
137: 
138:     def create_weights(
139:         self,
140:         layer: torch.nn.Module,
141:         num_experts: int,
142:         hidden_size: int,
143:         intermediate_size_per_partition: int,
144:         params_dtype: torch.dtype,
145:         **extra_weight_attrs,
146:     ):
147:         self.num_experts = num_experts
148:         weight_dtype = torch.uint8
149:         scale_dtype = torch.uint8
150:         mxfp4_block = 32
151: 
152:         layer.params_dtype = params_dtype
153:         layer.num_experts = num_experts
154:         self.intermediate_size = intermediate_size_per_partition
155:         self.hidden_size = hidden_size
156: 
157:         # Fused gate_up_proj (column parallel)
158:         w13_weight = torch.nn.Parameter(
159:             torch.zeros(
160:                 num_experts,
161:                 2 * intermediate_size_per_partition,
162:                 hidden_size // 2,
163:                 dtype=weight_dtype,
164:             ),
165:             requires_grad=False,
166:         )
167:         layer.register_parameter("w13_weight", w13_weight)
168:         set_weight_attrs(w13_weight, extra_weight_attrs)
169: 
170:         w13_weight_scale = torch.nn.Parameter(
171:             torch.zeros(
172:                 num_experts,
173:                 2 * intermediate_size_per_partition,
174:                 hidden_size // mxfp4_block,
175:                 dtype=scale_dtype,
176:             ),
177:             requires_grad=False,
178:         )
179:         layer.register_parameter("w13_weight_scale", w13_weight_scale)
180:         set_weight_attrs(w13_weight_scale, extra_weight_attrs)
181: 
182:         # down_proj (row parallel)
183:         w2_weight = torch.nn.Parameter(
184:             torch.zeros(
185:                 num_experts,
186:                 hidden_size,
187:                 intermediate_size_per_partition // 2,
188:                 dtype=weight_dtype,
189:             ),
190:             requires_grad=False,
191:         )
192:         layer.register_parameter("w2_weight", w2_weight)
193:         set_weight_attrs(w2_weight, extra_weight_attrs)
194: 
195:         w2_weight_scale = torch.nn.Parameter(
196:             torch.zeros(
197:                 num_experts,
198:                 hidden_size,
199:                 intermediate_size_per_partition // mxfp4_block,
200:                 dtype=scale_dtype,
201:             ),
202:             requires_grad=False,
203:         )
204:         layer.register_parameter("w2_weight_scale", w2_weight_scale)
205:         set_weight_attrs(w2_weight_scale, extra_weight_attrs)
206: 
207:         if self.moe.has_bias:
208:             w13_bias = torch.nn.Parameter(
209:                 torch.zeros(
210:                     num_experts,
211:                     2 * intermediate_size_per_partition,
212:                     dtype=torch.bfloat16,
213:                 ),
214:                 requires_grad=False,
215:             )
216:             layer.register_parameter("w13_bias", w13_bias)
217:             set_weight_attrs(w13_bias, extra_weight_attrs)
218: 
219:             w2_bias = torch.nn.Parameter(
220:                 torch.zeros(
221:                     num_experts,
222:                     hidden_size,
223:                     dtype=torch.bfloat16,
224:                 ),
225:                 requires_grad=False,
226:             )
227:             layer.register_parameter("w2_bias", w2_bias)
228:             set_weight_attrs(w2_bias, extra_weight_attrs)
229: 
230:     def _setup_kernel(
231:         self,
232:         layer: FusedMoE,
233:         w13: torch.Tensor,
234:         w2: torch.Tensor,
235:         w13_scale: torch.Tensor,
236:         w2_scale: torch.Tensor,
237:         w13_bias: torch.Tensor | None = None,
238:         w2_bias: torch.Tensor | None = None,
239:     ) -> None:
240:         num_experts = self.num_experts
241:         intermediate_size = self.intermediate_size
242:         hidden_size = self.hidden_size
243:         sf_block_size = 32
244: 
245:         # Shape assertions
246:         assert (
247:             w13.dim() == 3
248:             and w13.shape[0] == num_experts
249:             and w13.shape[1] == intermediate_size * 2

(Showing lines 50-249 of 413. Use offset=250 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/fp8.py",
  "offset": 1,
  "limit": 120
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/fp8.py</path>
<type>file</type>
<content>
1: # SPDX-License-Identifier: Apache-2.0
2: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
3: 
4: from typing import TYPE_CHECKING, Any
5: 
6: import torch
7: from torch.nn import Module
8: from torch.utils._python_dispatch import TorchDispatchMode
9: 
10: import vllm.envs as envs
11: import vllm.model_executor.layers.fused_moe.modular_kernel as mk
12: from vllm import _custom_ops as ops
13: from vllm._aiter_ops import rocm_aiter_ops
14: from vllm.distributed import get_tensor_model_parallel_world_size
15: from vllm.logger import init_logger
16: from vllm.model_executor.kernels.linear import (
17:     init_fp8_linear_kernel,
18: )
19: from vllm.model_executor.kernels.linear.scaled_mm import MarlinFP8ScaledMMLinearKernel
20: from vllm.model_executor.kernels.linear.scaled_mm import MarlinFP8ScaledMMLinearKernel
21: from vllm.model_executor.layers.attention import Attention
22: from vllm.model_executor.layers.fused_moe import (
23:     FusedMoE,
24:     FusedMoEMethodBase,
25:     FusedMoeWeightScaleSupported,
26: )
27: from vllm.model_executor.layers.fused_moe.config import (
28:     FusedMoEQuantConfig,
29: )
30: from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
31: from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
32:     convert_to_fp8_moe_kernel_format,
33:     make_fp8_moe_kernel,
34:     make_fp8_moe_quant_config,
35:     select_fp8_moe_backend,
36: )
37: from vllm.model_executor.layers.linear import (
38:     LinearBase,
39:     LinearMethodBase,
40:     UnquantizedLinearMethod,
41: )
42: from vllm.model_executor.layers.quantization import QuantizationMethods
43: from vllm.model_executor.layers.quantization.base_config import (
44:     QuantizationConfig,
45:     QuantizeMethodBase,
46: )
47: from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
48: from vllm.model_executor.layers.quantization.utils.fp8_utils import (
49:     W8A8BlockFp8LinearOp,
50:     create_fp8_input_scale,
51:     create_fp8_scale_parameter,
52:     create_fp8_weight_parameter,
53:     maybe_post_process_fp8_weight_block,
54:     process_fp8_input_tensor_strategy_moe,
55:     process_fp8_weight_block_strategy,
56:     process_fp8_weight_tensor_strategy,
57:     process_fp8_weight_tensor_strategy_moe,
58:     validate_fp8_block_shape,
59: )
60: from vllm.model_executor.layers.quantization.utils.marlin_utils import (
61:     get_marlin_input_dtype,
62: )
63: from vllm.model_executor.layers.quantization.utils.quant_utils import (
64:     GroupShape,
65:     is_layer_skipped,
66:     kFp8Dynamic128Sym,
67:     kFp8DynamicTensorSym,
68:     kFp8DynamicTokenSym,
69:     kFp8Static128BlockSym,
70:     kFp8StaticTensorSym,
71: )
72: from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
73:     cutlass_block_fp8_supported,
74:     cutlass_fp8_supported,
75:     normalize_e4m3fn_to_e4m3fnuz,
76: )
77: from vllm.model_executor.model_loader.reload.layerwise import (
78:     initialize_online_processing,
79: )
80: from vllm.model_executor.parameter import (
81:     BlockQuantScaleParameter,
82:     ModelWeightParameter,
83:     PerTensorScaleParameter,
84: )
85: from vllm.model_executor.utils import replace_parameter, set_weight_attrs
86: from vllm.platforms import current_platform
87: from vllm.utils.deep_gemm import (
88:     is_deep_gemm_supported,
89: )
90: 
91: if TYPE_CHECKING:
92:     from vllm.model_executor.models.utils import WeightsMapper
93: 
94: ACTIVATION_SCHEMES = ["static", "dynamic"]
95: 
96: logger = init_logger(__name__)
97: 
98: 
99: class Fp8Config(QuantizationConfig):
100:     """Config class for FP8."""
101: 
102:     def __init__(
103:         self,
104:         is_checkpoint_fp8_serialized: bool = False,
105:         activation_scheme: str = "dynamic",
106:         ignored_layers: list[str] | None = None,
107:         weight_block_size: list[int] | None = None,
108:     ) -> None:
109:         super().__init__()
110: 
111:         self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
112: 
113:         if activation_scheme not in ACTIVATION_SCHEMES:
114:             raise ValueError(f"Unsupported activation scheme {activation_scheme}")
115:         self.activation_scheme = activation_scheme
116:         self.ignored_layers = ignored_layers or []
117:         if weight_block_size is not None:
118:             if not is_checkpoint_fp8_serialized:
119:                 raise ValueError(
120:                     "The block-wise quantization only supports fp8-serialized "

(Showing lines 1-120 of 1070. Use offset=121 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/registry.py",
  "offset": 85,
  "limit": 30
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/model_executor/models/registry.py</path>
<type>file</type>
<content>
85:     "BailingMoeV2_5ForCausalLM": ("bailing_moe_linear", "BailingMoeV25ForCausalLM"),
86:     "BambaForCausalLM": ("bamba", "BambaForCausalLM"),
87:     "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
88:     "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
89:     "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
90:     "CohereForCausalLM": ("commandr", "CohereForCausalLM"),
91:     "Cohere2ForCausalLM": ("commandr", "CohereForCausalLM"),
92:     "CwmForCausalLM": ("llama", "LlamaForCausalLM"),
93:     "DbrxForCausalLM": ("dbrx", "DbrxForCausalLM"),
94:     "DeciLMForCausalLM": ("nemotron_nas", "DeciLMForCausalLM"),
95:     "DeepseekForCausalLM": ("deepseek_v2", "DeepseekForCausalLM"),
96:     "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
97:     "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
98:     "DeepseekV32ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
99:     "Dots1ForCausalLM": ("dots1", "Dots1ForCausalLM"),
100:     "Ernie4_5ForCausalLM": ("ernie45", "Ernie4_5ForCausalLM"),
101:     "Ernie4_5_MoeForCausalLM": ("ernie45_moe", "Ernie4_5_MoeForCausalLM"),
102:     "ExaoneForCausalLM": ("exaone", "ExaoneForCausalLM"),
103:     "Exaone4ForCausalLM": ("exaone4", "Exaone4ForCausalLM"),
104:     "ExaoneMoEForCausalLM": ("exaone_moe", "ExaoneMoeForCausalLM"),
105:     "Fairseq2LlamaForCausalLM": ("fairseq2_llama", "Fairseq2LlamaForCausalLM"),
106:     "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
107:     "FalconMambaForCausalLM": ("mamba", "MambaForCausalLM"),
108:     "FalconH1ForCausalLM": ("falcon_h1", "FalconH1ForCausalLM"),
109:     "FlexOlmoForCausalLM": ("flex_olmo", "FlexOlmoForCausalLM"),
110:     "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
111:     "Gemma2ForCausalLM": ("gemma2", "Gemma2ForCausalLM"),
112:     "Gemma3ForCausalLM": ("gemma3", "Gemma3ForCausalLM"),
113:     "Gemma3nForCausalLM": ("gemma3n", "Gemma3nForCausalLM"),
114:     "Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),

(Showing lines 85-114 of 1339. Use offset=115 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4 Pro · 13.0s)

_Thinking:_

Now I have everything I need. Let me also check:
1. The `override_quantization_method` signature in `Fp8Config` - does it accept `hf_config`?
2. The `config/model.py` call to `override_quantization_method` - does it pass `hf_config`?

From the model.py we can see:
```python
quantization_override = method.override_quantization_method(
    quant_cfg, self.quantization
)
```

This only passes 2 args: `quant_cfg` and `self.quantization`. But the upstream `DeepseekV4FP8Config.override_quantization_method()` expects 4 args including `hf_config`:
```python
def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None) -> QuantizationMethods | None:
```

And it checks `hf_config.model_type`. So the local `_verify_quantization()` would need to be updated to pass `hf_config` too.

Let me check the fp8.py override_quantization_method signature in the local codebase.

**Tool: grep**

**Input:**
```json
{
  "pattern": "def override_quantization_method",
  "path": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization"
}
```

**Output:**
```
Found 11 matches
/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/moe_wna16.py:
  Line 132:     def override_quantization_method(


/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/modelopt.py:
  Line 410:     def override_quantization_method(

  Line 1015:     def override_quantization_method(

  Line 1504:     def override_quantization_method(

  Line 2040:     def override_quantization_method(


/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/gguf.py:
  Line 87:     def override_quantization_method(


/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/gptq_marlin.py:
  Line 216:     def override_quantization_method(


/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/cpu_wna16.py:
  Line 106:     def override_quantization_method(


/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/inc.py:
  Line 455:     def override_quantization_method(


/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/base_config.py:
  Line 112:     def override_quantization_method(


/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/awq_marlin.py:
  Line 234:     def override_quantization_method(

```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/fp8.py",
  "offset": 121,
  "limit": 80
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/fp8.py</path>
<type>file</type>
<content>
121:                     "checkpoint for now."
122:                 )
123:             if len(weight_block_size) != 2:
124:                 raise ValueError(
125:                     "The quantization block size of weight must have 2 "
126:                     f"dimensions, but got {len(weight_block_size)} dimensions"
127:                 )
128:             if activation_scheme != "dynamic":
129:                 raise ValueError(
130:                     "The block-wise quantization only supports "
131:                     "dynamic activation scheme for now, but got "
132:                     f"{activation_scheme} activation scheme."
133:                 )
134:         self.weight_block_size = weight_block_size
135:         self.use_deep_gemm: bool | None = None
136: 
137:     @classmethod
138:     def get_name(cls) -> QuantizationMethods:
139:         return "fp8"
140: 
141:     @classmethod
142:     def get_supported_act_dtypes(cls) -> list[torch.dtype]:
143:         return [torch.bfloat16, torch.half]
144: 
145:     @classmethod
146:     def get_min_capability(cls) -> int:
147:         return 75
148: 
149:     @classmethod
150:     def get_config_filenames(cls) -> list[str]:
151:         return []
152: 
153:     def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
154:         if self.ignored_layers is not None:
155:             self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)
156: 
157:     @classmethod
158:     def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
159:         quant_method = cls.get_from_keys(config, ["quant_method"])
160:         is_checkpoint_fp8_serialized = "fp8" in quant_method
161:         activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
162:         ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
163:         weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
164:         if not ignored_layers:
165:             ignored_layers = cls.get_from_keys_or(
166:                 config, ["modules_to_not_convert"], None
167:             )
168:         return cls(
169:             is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
170:             activation_scheme=activation_scheme,
171:             ignored_layers=ignored_layers,
172:             weight_block_size=weight_block_size,
173:         )
174: 
175:     def get_quant_method(
176:         self, layer: torch.nn.Module, prefix: str
177:     ) -> "QuantizeMethodBase | None":
178:         if isinstance(layer, LinearBase):
179:             if is_layer_skipped(
180:                 prefix=prefix,
181:                 ignored_layers=self.ignored_layers,
182:                 fused_mapping=self.packed_modules_mapping,
183:             ):
184:                 return UnquantizedLinearMethod()
185:             if not self.is_checkpoint_fp8_serialized:
186:                 online_method = Fp8OnlineLinearMethod(self)
187:                 online_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
188:                 return online_method
189:             else:
190:                 offline_method = Fp8LinearMethod(self)
191:                 offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
192:                 return offline_method
193:         elif isinstance(layer, FusedMoE):
194:             if is_layer_skipped(
195:                 prefix=prefix,
196:                 ignored_layers=self.ignored_layers,
197:                 fused_mapping=self.packed_modules_mapping,
198:             ):
199:                 return UnquantizedFusedMoEMethod(layer.moe_config)
200:             if self.is_checkpoint_fp8_serialized:

(Showing lines 121-200 of 1070. Use offset=201 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4 Pro · 11.2s)

_Thinking:_

Good. Looking at the local codebase:

1. `Fp8Config` does NOT have an `override_quantization_method()` - it only has `from_config()` and `get_quant_method()`. The base `QuantizationConfig` has a default `override_quantization_method()` that returns `None`.

2. The `_verify_quantization()` in `config/model.py` calls `method.override_quantization_method(quant_cfg, self.quantization)` with only 2 args.

So for the `DeepseekV4FP8Config`, I need to:
1. Override `override_quantization_method()` - but the upstream version expects `hf_config` which the local code doesn't pass
2. Adapt it to work without `hf_config` by checking `quant_cfg` for distinguishing features

Alternative: I can modify `_verify_quantization()` to also pass `hf_config`, similar to the upstream change.

Let me check the base_config.py to see the current signature:

**Tool: read**

**Input:**
```json
{
  "offset": 100,
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/base_config.py",
  "limit": 30
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/base_config.py</path>
<type>file</type>
<content>
100:     @abstractmethod
101:     def get_config_filenames() -> list[str]:
102:         """List of filenames to search for in the model directory."""
103:         raise NotImplementedError
104: 
105:     @classmethod
106:     @abstractmethod
107:     def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
108:         """Create a config class from the model's quantization config."""
109:         raise NotImplementedError
110: 
111:     @classmethod
112:     def override_quantization_method(
113:         cls, hf_quant_cfg, user_quant
114:     ) -> QuantizationMethods | None:
115:         """
116:         Detects if this quantization method can support a given checkpoint
117:         format by overriding the user specified quantization method --
118:         this method should only be overwritten by subclasses in exceptional
119:         circumstances
120:         """
121:         return None
122: 
123:     @staticmethod
124:     def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
125:         """Get a value from the model's quantization config."""
126:         for key in keys:
127:             if key in config:
128:                 return config[key]
129:         raise ValueError(

(Showing lines 100-129 of 205. Use offset=130 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4 Pro · 10.1s)

_Thinking:_

OK so the base `override_quantization_method` has signature `(cls, hf_quant_cfg, user_quant)` with 2 params (besides cls). The upstream version added `hf_config=None` as a third param. This is backwards-compatible.

Now I have all the information. Let me implement:

1. Create `deepseek_v4_fp8.py` in the quantization directory with `DeepseekV4FP8Config`
2. Update `__init__.py` to register it
3. Update `config/model.py` to add it to overrides and pass `hf_config`

Actually, I need to be careful. Let me verify the local `config/model.py` call to `override_quantization_method` more carefully, and understand what `self.hf_config` or similar is available.

The `_verify_quantization` method has access to `self` which is a `ModelConfig`. Let me check if it has an `hf_config` attribute.

Let me search for hf_config in model.py.

**Tool: grep**

**Input:**
```json
{
  "pattern": "hf_config",
  "path": "/Users/ross/Documents/project/Lvllm/vllm/config/model.py"
}
```

**Output:**
```
Found 52 matches
/Users/ross/Documents/project/Lvllm/vllm/config/model.py:
  Line 48:     maybe_patch_hf_config_from_gguf,

  Line 158:     hf_config: PretrainedConfig = field(init=False)

  Line 161:     """The Hugging Face config of the text model (same as hf_config for text models)."""

  Line 162:     hf_config_path: str | None = None

  Line 350:             "hf_config_path",

  Line 466:         if isinstance(self.hf_config_path, str):

  Line 467:             self.hf_config_path = maybe_model_redirect(self.hf_config_path)

  Line 496:         hf_config = get_config(

  Line 497:             self.hf_config_path or self.model,

  Line 506:         hf_config = maybe_patch_hf_config_from_gguf(

  Line 508:             hf_config,

  Line 511:         self.hf_config = hf_config

  Line 513:             self._apply_dict_overrides(hf_config, dict_overrides)

  Line 514:         self.hf_text_config = get_hf_text_config(self.hf_config)

  Line 597:             self.hf_config,

  Line 692:             self.hf_config.model_type, ModelArchConfigConvertorBase

  Line 694:         convertor = convertor_cls(self.hf_config, self.hf_text_config)

  Line 737:         # If 'hf_config != hf_text_config' it's a nested config, i.e. multimodal

  Line 738:         cls += "MultiModal" if self.hf_config != self.hf_text_config else ""

  Line 1091:         if hasattr(self.hf_config, "dual_chunk_attention_config"):

  Line 1099:                 self.hf_config.dual_chunk_attention_config[

  Line 1104:                     not in self.hf_config.dual_chunk_attention_config

  Line 1106:                     self.hf_config.dual_chunk_attention_config[

  Line 1202:         if hasattr(self.hf_config, "is_mm_prefix_lm"):

  Line 1203:             return bool(self.hf_config.is_mm_prefix_lm)

  Line 1212:         if not hasattr(self.hf_config, "model_type"):

  Line 1214:         return self.hf_config.model_type in MM_PREFIX_LM_MODELS

  Line 1270:         # This function relies on 'layers_block_type' in hf_config,

  Line 1287:             block_configs = self.hf_config.block_configs

  Line 1305:             attn_type_list = getattr(self.hf_config, "attn_type_list", None)

  Line 1330:                     "attn_type_list, or a layer_types in the hf_config, "

  Line 1378:                 self.hf_config_path or self.model,

  Line 1490:         return is_encoder_decoder(self.hf_config)

  Line 1517:         return uses_mrope(self.hf_config)

  Line 1521:         return uses_xdrope_dim(self.hf_config)

  Line 1568:         layer_types = getattr(self.hf_config, "layer_types", None)

  Line 1591:         return bool(getattr(self.hf_config, "matryoshka_dimensions", None)) or getattr(

  Line 1592:             self.hf_config, "is_matryoshka", False

  Line 1597:         return getattr(self.hf_config, "matryoshka_dimensions", None)

  Line 1604:         use_pad_token = getattr(self.hf_config, "use_pad_token", None)

  Line 1611:         return getattr(self.hf_config, "use_sep_token", True)

  Line 1627:             config=self.hf_config, dtype=self.dtype, runner_type=self.runner_type

  Line 1653:         override = getattr(self.hf_config, "embedding_size", None)

  Line 1667:             and getattr(self.hf_config, "position_embedding_type", "") == "absolute"

  Line 1675:             hf_config=self.hf_text_config,

  Line 1695:                 is_causal = getattr(self.hf_config, "is_causal", True)

  Line 1816:         return getattr(self.hf_config, "quantization_config", None) is not None

  Line 2052:     hf_config: PretrainedConfig,

  Line 2106:     rope_parameters = getattr(hf_config, "rope_parameters", None)

  Line 2112:     if rope_parameters is not None and "gemma3" not in hf_config.model_type:

  Line 2152:                     hf_config, "original_max_position_embeddings", derived_max_model_len

  Line 2165:         model_max_length = getattr(hf_config, "model_max_length", None)

```

---

## Assistant (Build · DeepSeek V4 Pro · 10.4s)

_Thinking:_

`self.hf_config` is available in `ModelConfig`. I can use it when calling `override_quantization_method`.

Now let me implement the three changes:

1. Create `deepseek_v4_fp8.py` - the new quantization config
2. Update `__init__.py` - register the new method
3. Update `config/model.py` - add to overrides and pass `hf_config`

Let me start implementation. First, the new file.

Now I'll implement the quantization config. Let me create the three changes.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create deepseek_v4_fp8.py quantization config",
      "status": "in_progress",
      "priority": "high"
    },
    {
      "content": "Register deepseek_v4_fp8 in __init__.py",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Update config/model.py overrides and pass hf_config",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Register DeepseekV4ForCausalLM in model registry",
      "status": "pending",
      "priority": "medium"
    },
    {
      "content": "Verify the changes compile and are consistent",
      "status": "pending",
      "priority": "high"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create deepseek_v4_fp8.py quantization config",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Register deepseek_v4_fp8 in __init__.py",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Update config/model.py overrides and pass hf_config",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Register DeepseekV4ForCausalLM in model registry",
    "status": "pending",
    "priority": "medium"
  },
  {
    "content": "Verify the changes compile and are consistent",
    "status": "pending",
    "priority": "high"
  }
]
```

---

## Assistant (Build · DeepSeek V4 Pro · 15.0s)

_Thinking:_

Now let me create the deepseek_v4_fp8.py file. This is the key quantization config that:
1. Inherits from Fp8Config
2. Overrides `get_name()` to return `"deepseek_v4_fp8"`
3. Overrides `override_quantization_method()` to detect DeepSeek V4 checkpoints
4. Overrides `get_quant_method()` to route FusedMoE layers to MXFP4

The challenge is that the local codebase's `_verify_quantization()` only passes 2 args to `override_quantization_method()`. I need to:
- Either update `_verify_quantization()` to pass `hf_config` too
- Or make the config detect V4 without `hf_config`, using only `quant_cfg`

Looking at the quant_cfg for V4:
```json
{
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128]
}
```

The `fmt: "e4m3"` and `scale_fmt: "ue8m0"` are distinctive - they indicate E4M3 data with E8M0 scales (block-scaled FP8). But this could also match other models.

The safest approach is to update `_verify_quantization()` to pass `hf_config` too, similar to the upstream. But to minimize changes, I can just use `scale_fmt == "ue8m0"` as a heuristic for now, since that's the format DeepSeek V4 uses for its FP8 block quantization.

Actually, let me look at the modelopt.py override_quantization_method signatures more carefully to see if any need hf_config.

**Tool: read**

**Input:**
```json
{
  "offset": 410,
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/modelopt.py",
  "limit": 15
}
```

**Output:**
```
<path>/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/modelopt.py</path>
<type>file</type>
<content>
410:     def override_quantization_method(
411:         cls, hf_quant_cfg, user_quant
412:     ) -> QuantizationMethods | None:
413:         algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
414:         if algo is not None and algo == "FP8":
415:             return "modelopt"
416:         return None
417: 
418:     @classmethod
419:     def _from_config(
420:         cls,
421:         *,
422:         quant_method: str,
423:         kv_cache_quant_method: str | None,
424:         exclude_modules: list[str],

(Showing lines 410-424 of 2187. Use offset=425 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4 Pro · 20.6s)

_Thinking:_

OK, all the local `override_quantization_method` methods take only `(cls, hf_quant_cfg, user_quant)` - 2 params plus cls. The upstream added an optional `hf_config=None` param. To add this safely, I should:
1. Update `_verify_quantization()` to pass `hf_config=self.hf_config` as a keyword arg
2. This is backwards-compatible since existing overrides don't use it
3. My new `DeepseekV4FP8Config.override_quantization_method()` can accept `hf_config=None`

Let me implement all three files now.

**Tool: write**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/deepseek_v4_fp8.py",
  "content": "# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n\nimport torch\n\nfrom vllm.logger import init_logger\nfrom vllm.model_executor.layers.fused_moe import FusedMoE\nfrom vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod\nfrom vllm.model_executor.layers.quantization import QuantizationMethods\nfrom vllm.model_executor.layers.quantization.fp8 import Fp8Config\nfrom vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod\nfrom vllm.model_executor.layers.quantization.utils.quant_utils import (\n    is_layer_skipped,\n)\n\nlogger = init_logger(__name__)\n\n\nclass DeepseekV4FP8Config(Fp8Config):\n    \"\"\"FP8 config that routes MoE layers to MXFP4 quantization.\n\n    DeepSeek V4 checkpoints use FP8 for linear/attention layers but\n    MXFP4 for MoE expert weights. This config inherits standard FP8\n    behavior and overrides only the MoE dispatch.\n    \"\"\"\n\n    def __init__(self, *args, **kwargs):\n        super().__init__(*args, **kwargs)\n        self.is_scale_e8m0: bool = True\n\n    @classmethod\n    def get_name(cls) -> QuantizationMethods:\n        return \"deepseek_v4_fp8\"\n\n    @classmethod\n    def override_quantization_method(\n        cls, hf_quant_cfg, user_quant, hf_config=None\n    ) -> QuantizationMethods | None:\n        if not (\n            isinstance(hf_quant_cfg, dict)\n            and hf_quant_cfg.get(\"quant_method\") in (\"fp8\", \"deepseek_v4_fp8\")\n        ):\n            return None\n        model_type = getattr(hf_config, \"model_type\", None)\n        if model_type == \"deepseek_v4\" or user_quant == \"deepseek_v4_fp8\":\n            return \"deepseek_v4_fp8\"\n        return None\n\n    def get_quant_method(self, layer, prefix):\n        if isinstance(layer, FusedMoE):\n            if is_layer_skipped(\n                prefix=prefix,\n                ignored_layers=self.ignored_layers,\n                fused_mapping=self.packed_modules_mapping,\n            ):\n                return UnquantizedFusedMoEMethod(layer.moe_config)\n            return Mxfp4MoEMethod(layer.moe_config)\n        return super().get_quant_method(layer, prefix)\n\n    def is_mxfp4_quant(self, prefix, layer):\n        return isinstance(layer, FusedMoE)\n"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · DeepSeek V4 Pro · 6.1s)

_Thinking:_

Now update `__init__.py` to register the new quantization method.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create deepseek_v4_fp8.py quantization config",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Register deepseek_v4_fp8 in __init__.py",
      "priority": "high",
      "status": "in_progress"
    },
    {
      "content": "Update config/model.py overrides and pass hf_config",
      "priority": "high",
      "status": "pending"
    },
    {
      "content": "Register DeepseekV4ForCausalLM in model registry",
      "priority": "medium",
      "status": "pending"
    },
    {
      "content": "Verify the changes compile and are consistent",
      "priority": "high",
      "status": "pending"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create deepseek_v4_fp8.py quantization config",
    "priority": "high",
    "status": "completed"
  },
  {
    "content": "Register deepseek_v4_fp8 in __init__.py",
    "priority": "high",
    "status": "in_progress"
  },
  {
    "content": "Update config/model.py overrides and pass hf_config",
    "priority": "high",
    "status": "pending"
  },
  {
    "content": "Register DeepseekV4ForCausalLM in model registry",
    "priority": "medium",
    "status": "pending"
  },
  {
    "content": "Verify the changes compile and are consistent",
    "priority": "high",
    "status": "pending"
  }
]
```

---

## Assistant (Build · DeepSeek V4 Pro · 5.9s)

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/ross/Documents/project/Lvllm/vllm/model_executor/layers/quantization/__init__.py",
  "oldString": "    \"mxfp4\",\n    \"mxfp8\",",
  "newString": "    \"mxfp4\",\n    \"deepseek_v4_fp8\",\n    \"mxfp8\","
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · DeepSeek V4 Pro · 0.7s)

---

## User

continue

---

## Assistant (Build · DeepSeek V4 Pro · 0.7s)

---

