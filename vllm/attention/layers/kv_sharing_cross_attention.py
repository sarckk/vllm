# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import List, Optional

import torch

from vllm import envs
from vllm.attention import AttentionType
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig, QuantizationConfig
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata, make_local_attention_virtual_batches)

from .utils import create_custom_attention_backend
from ..layer import Attention


def compute_fast_prefill_attn_metadata(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    logits_indices = common_attn_metadata.logits_indices
    num_reqs = common_attn_metadata.num_reqs
    query_start_loc = common_attn_metadata.query_start_loc
    seq_lens = common_attn_metadata.seq_lens
    # Example inputs
    # num_reqs: 3
    # generation_indices:  [14, 18, 19, 27]
    # query_start_loc: [0, 15, 20, 28]
    # seq_lens:        [41, 31, 40]

    # Find how many decode indices belong to each request
    # request_ids: [0, 1, 1, 2]
    request_ids = torch.bucketize(logits_indices,
                                    query_start_loc[1:],
                                    right=True)

    # Figure out how many tokens are in each request
    # num_decode_tokens: [1, 2, 1]
    num_decode_tokens = torch.bincount(request_ids, minlength=num_reqs)

    # Calculate new query_start_loc with tokens in generation_indices
    # decode_query_start_loc: [0, 1, 3, 4]
    decode_query_start_loc = torch.empty(num_reqs + 1,
                                            device=query_start_loc.device,
                                            dtype=query_start_loc.dtype)

    decode_query_start_loc[0] = 0
    decode_query_start_loc[1:] = torch.cumsum(num_decode_tokens, dim=0)
    decode_max_query_len = int(num_decode_tokens.max().item())
    total_num_decode_tokens = int(num_decode_tokens.sum().item())

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=decode_query_start_loc,
        # TODO: optimize
        query_start_loc_cpu=decode_query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=total_num_decode_tokens,
        max_query_len=decode_max_query_len,
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        logits_indices=logits_indices,
        causal=True,
    )
    return common_attn_metadata

class KVSharingCrossAttention(Attention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        use_mla: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        attn_backend: Optional[type[AttentionBackend]] = None,
        **extra_impl_args,
    ):
        dtype = torch.get_default_dtype()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16
        
        attention_chunk_size = \
            extra_impl_args.get("attention_chunk_size", None)

        if envs.VLLM_USE_V1:
            underlying_attn_backend = get_attn_backend(head_size, dtype,
                                                       kv_cache_dtype,
                                                       block_size)

            attn_metadata_builder_prefix = "KVSharing"
            # TODO: check if fast prefill optimiztion path enabled
            # 1) user guarantees that this is a valid fast prefill path 
            # 2) Check that all requests on current iteration are decode?
            #    Maybe by checking that length of tokens for each req all one?
            fast_prefill = (
                cache_config is not None 
                and cache_config.kv_sharing_fast_prefill
            )

            if fast_prefill:
                attn_metadata_builder_prefix += "FastPrefill"
            if attention_chunk_size is not None:
                attn_metadata_builder_prefix += "ChunkedLocalAttention"

            def build_preprocess_fn(cm: CommonAttentionMetadata):
                preprocessed_cm = cm

                if fast_prefill:
                    preprocessed_cm = compute_fast_prefill_attn_metadata(cm)

                if attention_chunk_size is not None:
                    preprocessed_cm = make_local_attention_virtual_batches(attention_chunk_size, cm, block_size)
                return preprocessed_cm

            attn_backend = create_custom_attention_backend(
                attn_metadata_builder_prefix, underlying_attn_backend, build_preprocess_fn)
        else:
            attn_backend = None

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            per_layer_sliding_window=per_layer_sliding_window,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=attn_backend,
        )
