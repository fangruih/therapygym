"""
LLM handler factory for the exact-style bench runner.

This maps the bench runner `ModelType` to public provider handlers or local vLLM.
"""

import os

from therapygym.rllm.llm_api.anthropic_handler import AnthropicHandler
from therapygym.rllm.llm_api.openai_handler import OpenAIHandler
from therapygym.rllm.llm_api.vllm_handler import VLLMHandler

from .model_configs import ModelType


def get_llm_handler(model_type: ModelType):
    # Human is handled elsewhere (no API call needed)
    if model_type == ModelType.HUMAN:
        return None

    # Local vLLM (OpenAI-compatible)
    if model_type == ModelType.QWEN3_4B_LOCAL:
        host = os.getenv("VLLM_HOST", "localhost")
        port = int(os.getenv("VLLM_PORT", "8000"))
        return VLLMHandler(host=host, port=port)

    if model_type == ModelType.CLAUDE_3_7:
        return AnthropicHandler(model="CLAUDE_3_7_ANTHROPIC")
    if model_type == ModelType.O3_MINI:
        return OpenAIHandler(model_type=model_type, model_name="o3-mini")
    if model_type == ModelType.O1:
        return OpenAIHandler(model_type=model_type, model_name="o1")
    raise NotImplementedError(
        f"{model_type.name} requires a provider-specific adapter that is not included in the public release."
    )

