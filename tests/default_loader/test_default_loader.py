import pytest
from vllm import SamplingParams

model_ref = "Qwen/Qwen2.5-VL-3B-Instruct"
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, seed=0)

@pytest.mark.parametrize("model", [
    "Qwen/Qwen2.5-VL-3B-Instruct",
])
def test_load_mm_params_skipped(vllm_runner, model):
    with vllm_runner(
        model,
        task="generate",
        max_model_len=8192,
        enforce_eager=True,
    ) as ref_model:
        ref_outputs = ref_model.generate(prompts, sampling_params)

    with vllm_runner(
        model,
        task="generate",
        max_model_len=8192,
        # Set MM limit to 0 to skip loading params related to MM params
        limit_mm_per_prompt={
            "image": 0,
            "video": 0,
        },
        enforce_eager=True,
    ) as mm_load_skipped_model:
        mm_load_skipped_outputs = mm_load_skipped_model.generate(
            prompts, sampling_params)

        assert ref_outputs == mm_load_skipped_outputs
