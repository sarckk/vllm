# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import random

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel

from ...utils import fork_new_process_for_each_test


@pytest.fixture
def test_prompts():
    """
    Adapted from tests/v1/e2e/test_spec_decode.py
    """
    prompt_types = ["repeat", "sentence"]
    # Setting higher num prompts increases the chance of numerics mismatch
    # due to matrix multiplication numerics depending on batch dimension
    num_prompts = 10
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        if kind == "repeat":
            prompt = f"""please repeat the word '{word}' 10 times."""
        elif kind == "sentence":
            prompt = f"""please give a ten-word sentence that
            uses the word {word} at least once."""
        else:
            raise ValueError(f"Unknown prompt type: {kind}")
        prompts.append(prompt)

    return prompts


@fork_new_process_for_each_test
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_kv_sharing_fast_prefill(
    monkeypatch: pytest.MonkeyPatch,
    enforce_eager: bool,
    test_prompts: list[str],
):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    compilation_config = CompilationConfig(
        # This allows vLLM compilation backend to handle allocating and
        # managing buffers for cudagraph
        cudagraph_copy_inputs=True,
        level=CompilationLevel.PIECEWISE
        if not enforce_eager else CompilationLevel.NO_COMPILATION)

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        llm = LLM(
            model="google/gemma-3n-E2B-it",
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
        )
        ref_responses = llm.generate(test_prompts, sampling_params)

        del llm
        gc.collect()
        torch.cuda.empty_cache()

        llm = LLM(model="google/gemma-3n-E2B-it",
                  enforce_eager=enforce_eager,
                  compilation_config=compilation_config,
                  kv_sharing_fast_prefill=True)
        optimized_responses = llm.generate(test_prompts, sampling_params)

        misses = 0

        for ref_response, optimized_response in zip(ref_responses,
                                                    optimized_responses):
            if ref_response.outputs[0].text != optimized_response.outputs[
                    0].text:
                misses += 1

        assert misses == 0
