# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import get_args

import vllm.envs as envs
from vllm.config import AttentionConfig
from vllm.config.model import _call_quantization_override
from vllm.config.speculative import MTPModelTypes, SpeculativeConfig
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)
from vllm.model_executor.models.deepseek_v4 import DeepseekV4FP8Config


def test_deepseek_v4_hf_config_override_sets_v4_mtp_architecture():
    hf_config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_nextn_predict_layers=3,
    )

    overridden = SpeculativeConfig.hf_config_override(hf_config)

    assert overridden is hf_config
    assert hf_config.model_type == "deepseek_v4_mtp"
    assert hf_config.architectures == ["DeepSeekV4MTPModel"]
    assert hf_config.n_predict == 3
    assert "deepseek_v4_mtp" in get_args(MTPModelTypes)


def test_deepseek_v4_mtp_arch_convertor_uses_mla_shape(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MLA_DISABLE", False)

    hf_config = DeepseekV4Config(
        architectures=["DeepSeekV4MTPModel"],
        num_hidden_layers=61,
        num_nextn_predict_layers=2,
        num_attention_heads=128,
        hidden_size=7168,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    hf_config.model_type = "deepseek_v4_mtp"

    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS["deepseek_v4_mtp"]
    convertor = convertor_cls(hf_config, hf_config)

    assert convertor.is_deepseek_mla() is True
    assert convertor.get_num_hidden_layers() == 2
    assert convertor.get_head_size() == 576


def test_quant_override_helper_supports_legacy_signature():
    class LegacyQuantConfig:
        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant):
            assert hf_quant_cfg["quant_method"] == "fp8"
            assert user_quant is None
            return "legacy_quant"

    override = _call_quantization_override(
        LegacyQuantConfig,
        {"quant_method": "fp8"},
        None,
        SimpleNamespace(model_type="deepseek_v4"),
    )

    assert override == "legacy_quant"


def test_quant_override_helper_passes_hf_config_to_deepseek_v4():
    override = _call_quantization_override(
        DeepseekV4FP8Config,
        {"quant_method": "fp8"},
        None,
        SimpleNamespace(model_type="deepseek_v4"),
    )

    assert override == "deepseek_v4_fp8"


def test_attention_config_exposes_fp4_indexer_cache_flag():
    attention_config = AttentionConfig()

    assert attention_config.use_fp4_indexer_cache is False
