from typing import Callable
import functools
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata, subclass_attention_backend, subclass_attention_metadata_builder)


@functools.lru_cache
def create_custom_attention_backend(
    prefix: str,
    underlying_attn_backend: AttentionBackend,
    build_preprocess_fn: Callable[[CommonAttentionMetadata],
                                  CommonAttentionMetadata],
) -> type[AttentionBackend]:
    # Dynamically create a new attention backend that wraps the
    # underlying attention backend but applies
    # `build_preproces_fn` before calling `build(...)`
    builder_cls = subclass_attention_metadata_builder(
        name_prefix=prefix,
        builder_cls=underlying_attn_backend.get_builder_cls(),
        build_preprocess_fn=build_preprocess_fn)
    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=builder_cls)

    return attn_backend
