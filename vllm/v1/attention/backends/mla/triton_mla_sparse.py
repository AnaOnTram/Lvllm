# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120-safe Triton sparse MLA backend."""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseImpl,
)


class TritonMLASparseBackend(DeepseekV4FlashMLASparseBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8_ds_mla",
        "fp8",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type[FlashMLASparseImpl]:
        # DeepSeek V4 has its own attention layer and dispatches the Triton
        # kernels directly. Returning the existing sparse impl keeps generic
        # backend plumbing satisfied for paths that ask for an impl class.
        return FlashMLASparseImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True
