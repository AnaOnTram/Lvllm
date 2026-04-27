# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization import mxfp4
from vllm.model_executor.offloader.uva import UVAOffloader


class _DummyLayer(torch.nn.Module):
    def __init__(self, layer_name: str):
        super().__init__()
        self.layer_name = layer_name


class _FakeTensor:
    def __init__(self, device_type: str, numel: int, element_size: int):
        self.device = torch.device(device_type)
        self._numel = numel
        self._element_size = element_size

    def to(self, device: str | torch.device):
        return _FakeTensor(str(torch.device(device)), self._numel, self._element_size)

    def pin_memory(self):
        return self

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class _FakeParam:
    def __init__(self, data: _FakeTensor):
        self.data = data
        self._vllm_is_uva_offloaded = True

    @property
    def device(self) -> torch.device:
        return self.data.device


class _FakeModule:
    def __init__(self, param_name: str, param: _FakeParam):
        self._param_name = param_name
        self._param = param

    def parameters(self):
        yield self._param

    def named_parameters(self):
        yield self._param_name, self._param


def test_matches_offload_segments_uses_full_segment_matching():
    full_name = "model.layers.0.ffn.experts.w13_weight"
    assert mxfp4._matches_offload_segments(full_name, {"experts"})
    assert mxfp4._matches_offload_segments(full_name, {"experts.w13_weight"})
    assert not mxfp4._matches_offload_segments(full_name, {"expert"})
    assert not mxfp4._matches_offload_segments(full_name, {"shared_experts"})


def test_make_moe_parameter_allocates_direct_uva_storage(monkeypatch):
    layer = _DummyLayer("model.layers.0.ffn.experts")
    offloader = UVAOffloader(
        cpu_offload_max_bytes=1024,
        cpu_offload_params={"experts"},
    )
    offloader.pin_memory = False
    offloader.uva_offloading = True

    monkeypatch.setattr(mxfp4, "get_offloader", lambda: offloader)
    monkeypatch.setattr(
        mxfp4,
        "get_accelerator_view_from_cpu_tensor",
        lambda cpu_tensor: cpu_tensor,
    )

    param = mxfp4._make_moe_parameter(layer, "w13_weight", (2, 3), torch.uint8)

    assert param.device.type == "cpu"
    assert getattr(param, "_vllm_is_uva_offloaded", False)
    assert offloader.cpu_offload_bytes == 6


def test_uva_offloader_skips_params_already_backed_by_uva(monkeypatch):
    offloader = UVAOffloader(
        cpu_offload_max_bytes=1024,
        cpu_offload_params={"experts"},
    )
    offloader.pin_memory = False
    offloader.uva_offloading = True

    param = _FakeParam(_FakeTensor("cuda", 8, 1))
    module = _FakeModule("ffn.experts.w13_weight", param)

    offloader._maybe_offload_to_cpu(module)

    assert offloader.cpu_offload_bytes == 0
    assert getattr(param, "_vllm_is_uva_offloaded", False)
    assert param.device.type == "cuda"
