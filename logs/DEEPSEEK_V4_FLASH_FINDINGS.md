# DeepSeek V4 Flash Support Findings And Porting Plan

## Reader And Goal

This document is for the next engineer or model working on DeepSeek V4 Flash support in this fork.

After reading it, the reader should be able to choose the correct implementation strategy and execute the port in the right order without repeating the earlier partial-quantization detour.

## Executive Summary

DeepSeek V4 Flash support is not blocked by a missing generic MXFP4 implementation.

The real gap is that this fork does not yet contain the DeepSeek V4 model stack. **IMPORTANT CORRECTION (Phase 0):** As of 2026-04-27, upstream vLLM main does NOT have DeepSeek V4 support. The implementation lives in open PR [#40760](https://github.com/vllm-project/vllm/pull/40760) (`zyongye/vllm:dsv4` branch), which is still under review. Previous assumptions that V4 had been merged into upstream were incorrect.

The port must source from the PR branch, not upstream main. A greenfield local rewrite is likely to be slower and riskier.

## Findings

### 1. Generic MXFP4 support already exists in this fork

This codebase already has core MXFP4 support for MoE execution. That means "the project does not support MXFP4 at all" is not the right diagnosis.

What exists already:

- A built-in `mxfp4` quantization mode
- Existing MXFP4 MoE backends
- Existing MXFP4 utilities and tests

What does not follow from that:

- It does not mean DeepSeek V4 Flash is already supported
- It does not mean a single new quantization alias is enough
- It does not mean the model can load through the current DeepSeek V2/V3 path

### 2. DeepSeek V4 Flash uses a custom mixed quantization path

The earlier investigation found that upstream uses a dedicated quantization mode named `deepseek_v4_fp8`.

The important behavior is:

- Dense and attention paths use FP8-style handling
- MoE expert weights route to MXFP4 handling

That is why a DeepSeek V4 port cannot be reduced to "just use mxfp4" or "just use fp8". The model needs the custom dispatch rules as part of its own implementation.

### 3. This fork is missing the model architecture, not just the quantization hook

This fork currently has DeepSeek V2 and V3 support, but not the later DeepSeek V4 stack.

The earlier session concluded that upstream already added DeepSeek V4 support after this fork's current baseline. The missing work is not just one config class. It includes model architecture, attention/compression pieces, model registration, parser/tokenizer integrations, and specialized kernels.

### 4. The earlier DeepSeek edit was only a partial patch

The reverted attempt added a new quantization name and a draft quantization config, but it did not complete the full wiring.

The partial attempt was insufficient because:

- The quantization registry change was incomplete
- Quantization auto-detection was not fully updated
- The DeepSeek V4 model class was still absent
- The supporting attention and kernel stack was still absent

The correct lesson is not "the idea was close". The correct lesson is "this work has to be ported as a complete feature slice".

### 5. 16 GB VRAM is a deployment constraint, not the first porting task

The target machine has 192 GB RAM and 16 GB VRAM. That matters, but it should not drive the first coding step.

The first goal is feature correctness:

- make the model architecture load
- make quantization resolve correctly
- make the specialized execution path run

Only after that should the fork focus on making the model usable on a 16 GB GPU with offload and memory tuning.

## What The Next Implementation Should Not Do

- Do not restart from a tiny "quantization-only" patch
- Do not assume DeepSeek V4 can be mapped onto the existing DeepSeek V2 class with a few conditionals
- Do not treat MXFP4 support as the main missing piece
- Do not optimize for low VRAM before the model loads and runs correctly
- Do not split the port into isolated edits that leave the repo half-wired

## Recommended Strategy

Port the upstream DeepSeek V4 support as a coherent bundle, then adapt and verify it in this fork.

That means the work should follow this order:

1. Land the model architecture and its immediate dependencies
2. Land the custom DeepSeek V4 quantization routing
3. Land the registration and auto-detection wiring
4. Land the specialized kernels and custom ops required by the model path
5. Add tests for load, quantization dispatch, and inference
6. Only then tune for the 16 GB VRAM target

## Strong TODO List

### Phase 0: Revalidate upstream baseline (COMPLETED 2026-04-27)

- [x] Run the duplicate-work checks required by `AGENTS.md` before proposing any PR
- [x] Identify the exact upstream DeepSeek V4 support change set that should be ported
- [x] Confirm whether this fork diverged in any model-executor, attention, or quantization interfaces that will require adaptation during the port
- [x] Decide whether the work should be cherry-picked in chunks or copied as a manual port

**Phase 0 Results:**

1. **Upstream V4 status:** PR [#40760](https://github.com/vllm-project/vllm/pull/40760) "Support DeepseekV4" is **STILL OPEN** (not merged) on `zyongye/vllm:dsv4` branch. 97 files, +16,968/-760 lines. `gh pr list --state merged --search "deepseek v4"` shows only bugfix PRs, not the implementation PR.

2. **Port source:** Must use `zyongye/vllm:dsv4` branch, not upstream main.

3. **Fork divergence:** Fork merged from upstream at `0a40cd2d9` (April 6, 2026). Since then, 8 local commits modifying 8 files (README, config, FLA kernel fixes, MoE runner optimization, GPU memory fixes).

4. **Overlap analysis — files modified by both fork and V4 PR (will need merge):**
   - `vllm/model_executor/layers/fused_moe/layer.py` — fork has LK_POWER_SAVING changes; PR adds MegaMoE dispatch
   - `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` — fork has memory/reduce max_num_seqs changes; PR adds V4 support
   - `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` — fork has CPU backend fix; PR adds V4 expert mapping
   - `vllm/v1/worker/gpu_model_runner.py` — fork has NULL_BLOCK_ID/CUDA graph fixes; PR adds V4 model runner integration
   - `vllm/v1/attention/backends/utils.py` — fork has NULL_BLOCK_ID fix; PR adds V4 attention ops

5. **PR #40760 key new files to port:**
   - Model: `vllm/model_executor/models/deepseek_v4.py`, `deepseek_v4_mtp.py`
   - Attention: `vllm/model_executor/layers/deepseek_v4_attention.py` (1062 lines)
   - Compressor: `vllm/model_executor/layers/deepseek_compressor.py` (436 lines)
   - MHC: `vllm/model_executor/layers/mhc.py` (436 lines)
   - Config: `vllm/transformers_utils/configs/deepseek_v4.py`
   - Tokenizer: `vllm/tokenizers/deepseek_v4.py`, `deepseek_v4_encoding.py`
   - Renderer: `vllm/renderers/deepseek_v4.py`
   - Tool parser: `vllm/tool_parsers/deepseekv4_tool_parser.py`
   - Attention ops: `vllm/v1/attention/ops/deepseek_v4_ops/` (5 files)
   - CUDA kernels: `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`, `csrc/moe/topk_softplus_sqrt_kernels.cu`
   - Tests: 15+ test files

6. **Quantization:** The PR adds `"deepseek_v4_fp8"` to `QuantizationMethods` Literal. `DeepseekV4FP8Config` extends `Fp8Config`, routing MoE layers to `Mxfp4MoEMethod` and dense layers to standard FP8.

7. **Decision:** Manual port is the right approach. Cherry-picking from a non-merged PR branch is unreliable. Copy new files wholesale and manually merge changes to modified files, resolving the 5 overlapping files carefully.

### Phase 1: Port the model stack first

- [ ] Port the DeepSeek V4 model module itself
- [ ] Port the DeepSeek V4 attention path and any companion attention helpers
- [ ] Port the compressor and related KV-compression logic used by the model
- [ ] Port the manifold or hyper-connection support used by the new architecture
- [ ] Port any expert-parameter mapping helpers used by the model loader
- [ ] Port any custom ops wrappers that the forward path depends on

Exit condition:

- [ ] The fork contains the full DeepSeek V4 architecture path and it imports cleanly

### Phase 2: Port quantization as a model-specific feature

- [ ] Add the `deepseek_v4_fp8` quantization config exactly as part of the model support story, not as an isolated feature
- [ ] Make sure the config routes MoE layers to MXFP4 while keeping the dense path on FP8 behavior
- [ ] Update quantization auto-detection so DeepSeek V4 checkpoints resolve to the correct quantization mode
- [ ] Verify that the quantization detection path receives enough model metadata to distinguish DeepSeek V4 from generic FP8 checkpoints

Exit condition:

- [ ] A DeepSeek V4 checkpoint selects the correct quantization mode without manual patching

### Phase 3: Wire model registration and serving integrations

- [ ] Register the DeepSeek V4 architecture in the model registry
- [ ] Register any tokenizer-mode integration required by DeepSeek V4
- [ ] Register any tool parser or reasoning parser integration required by DeepSeek V4
- [ ] Verify that serving code can resolve the model type, architecture, and parser settings together

Exit condition:

- [ ] The serving stack recognizes DeepSeek V4 as a first-class model type

### Phase 4: Port the specialized execution pieces

- [ ] Port the specialized DeepSeek V4 fused kernels and any custom CUDA or Triton pieces they rely on
- [ ] Port any MegaMoE or DeepGEMM-related pieces required by the upstream implementation
- [ ] Verify that the custom ops are registered and reachable from the runtime path
- [ ] Verify that the fallback behavior is well-defined if a required backend is not available

Exit condition:

- [ ] The forward pass can execute without missing-op failures

### Phase 5: Add verification before optimization

- [ ] Add a minimal import test for the DeepSeek V4 model stack
- [ ] Add a quantization-selection test proving that DeepSeek V4 checkpoints resolve to `deepseek_v4_fp8`
- [ ] Add a model-load test that covers the expected checkpoint format
- [ ] Add at least one inference smoke test for the DeepSeek V4 serving path
- [ ] Add a regression test ensuring the generic FP8 and generic MXFP4 paths still behave as before

Exit condition:

- [ ] The feature is covered by tests that fail on missing registration, bad quantization routing, or missing runtime pieces

### Phase 6: Tune for the 16 GB VRAM target

- [ ] Measure real memory usage after the model can load and run
- [ ] Add or tune CPU offload for the 192 GB RAM target
- [ ] Reuse the existing NUMA interleave support when CPU-heavy offload paths are involved
- [ ] Evaluate expert parallelism and expert offload settings for the target machine
- [ ] Reduce max sequence length and other serving defaults to something realistic for a 16 GB GPU
- [ ] Document the final launch recipe that is practical for the target hardware

Exit condition:

- [ ] There is a reproducible serving configuration for the target machine, with known limits called out explicitly

## Suggested Validation Sequence

Use this order when testing:

1. Import-time validation
2. Registry and quantization-resolution validation
3. Model-load validation
4. Single-request inference smoke test
5. Multi-request serving stability test
6. Memory-tuning and offload validation on the target hardware

This order matters. If the model is not registered correctly or the wrong quantization path is selected, later memory or serving work will only hide the real failure.

## Practical Notes For The Next Attempt

- Treat the upstream DeepSeek V4 implementation as the source of truth
- Prefer a coherent port over a local re-interpretation
- Keep each phase testable before moving to the next
- If a partial port does not import, stop and finish that slice before adding more surface area
- Keep the low-VRAM objective in scope, but only after correctness is established

## Definition Of Done

DeepSeek V4 Flash support should only be considered complete when all of the following are true:

- The fork recognizes the model type and architecture
- Quantization auto-detection resolves to the DeepSeek V4-specific path
- The model loads without manual monkey-patching
- A serving smoke test completes successfully
- The implementation is covered by focused tests
- A documented configuration exists for the 192 GB RAM / 16 GB VRAM target, including its limits
