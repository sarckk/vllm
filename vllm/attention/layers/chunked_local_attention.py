# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import List, Optional

import torch

from vllm import envs
from vllm.attention.backends.abstract import get_attn_backend
from vllm.attention.backends.mla.common import (
    make_local_attention_virtual_batches)
from vllm.attention.backends.utils import (subclass_attention_backend,
                                           subclass_attention_metadata_builder)
from vllm.config import CacheConfig, QuantizationConfig

from ..layer import Attention


class ChunkedLocalAttention(Attention):

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 attention_chunk_size: int,
                 num_kv_heads: Optional[int] = None,
                 alibi_slopes: Optional[List[float]] = None,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        dtype = torch.get_default_dtype()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        # For v1 we have backend agnostic iRoPE (local chunked attention)
        # we have to store the flag on the layer so gpu model runner can
        # set KVSpec appropriately (and pop it so it doesnt get passed to
        # the backends)
        if envs.VLLM_USE_V1:
            underlying_attn_backend = get_attn_backend(head_size, dtype,
                                                       kv_cache_dtype,
                                                       block_size)

            # Dynamically create a new attention backend that wraps the
            # underlying attention backend but applies
            # `make_local_attention_virtual_batches` before calling `build(...)`
            builder_cls = subclass_attention_metadata_builder(
                name_prefix="ChunkedLocalAttention",
                builder_cls=underlying_attn_backend.get_builder_cls(),
                build_preprocess_fn=lambda
                cm: make_local_attention_virtual_batches(
                    attention_chunk_size, cm, block_size))
            attn_backend = subclass_attention_backend(
                name_prefix="ChunkedLocalAttention",
                attention_backend_cls=underlying_attn_backend,
                builder_cls=builder_cls)
        else:
            attn_backend = None

        super().__init__(num_heads=num_heads,
                         head_size=head_size,
                         scale=scale,
                         num_kv_heads=num_kv_heads,
                         alibi_slopes=alibi_slopes,
                         cache_config=cache_config,
                         quant_config=quant_config,
                         prefix=prefix,
                         attn_backend=attn_backend)
