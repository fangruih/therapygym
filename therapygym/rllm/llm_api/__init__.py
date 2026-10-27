from enum import Enum
from typing import Optional
from .vllm_handler import VLLMHandler
from .openai_handler import OpenAIHandler
from .anthropic_handler import AnthropicHandler
from .bedrock_handler import BedrockHandler
from .litellm_handler import LiteLLMHandler

class ModelType(Enum):
    """Supported model types for patient simulation."""
    O3_MINI = "O3_MINI"  # Original O3 mini model (SOM/legacy)
    QWEN_LOCAL = "QWEN_LOCAL"  # Local VLLM-served Qwen
    O3_MINI_OPENAI = "o3_mini_openai"  # Official OpenAI o3-mini via PAPI
    CLAUDE_3_7_ANTHROPIC = "claude_3_7_anthropic"
    CLAUDE_3_7_AMAZON = "claude_3_7_amazon"
    CLAUDE_3_5_HAIKU_ANTHROPIC = "claude_3_5_haiku_anthropic"
    # Generic provider router (requires `litellm` installed)
    LITELLM = "litellm"

def get_llm_handler(model_type: ModelType, host: str = "localhost", port: int = 8000):
    """Get LLM handler based on model type.
    
    Args:
        model_type: Type of model to use
        host: Host for VLLM server (only for QWEN_LOCAL)
        port: Port for VLLM server (only for QWEN_LOCAL)
    
    Returns:
        LLM handler instance
    """
    # Compare based on enum NAME to avoid confusion with value casing
    name = model_type.name
    if name == "O3_MINI":
        print("Using O3_MINI model (legacy/SOM)")
        # return SOMHandler(ModelType.O3_MINI)  # If needed, re-enable SOM integration
        raise NotImplementedError("O3_MINI via SOM is not enabled. Use O3_MINI_OPENAI for OpenAI PAPI.")
    elif name == "QWEN_LOCAL":
        print("Using QWEN_LOCAL model")
        from .vllm_handler import VLLMHandler
        return VLLMHandler(host=host, port=port)
    elif name == "O3_MINI_OPENAI":
        print("Using O3_MINI_OPENAI (OpenAI PAPI) model")
        from .openai_handler import OpenAIHandler
        return OpenAIHandler(model_type=model_type, model_name="o3-mini")
    elif name == "CLAUDE_3_7_ANTHROPIC":
        print("Using CLAUDE_3_7_ANTHROPIC model")
        from .anthropic_handler import AnthropicHandler
        return AnthropicHandler(model="CLAUDE_3_7_ANTHROPIC")
    elif name == "CLAUDE_3_5_HAIKU_ANTHROPIC":
        print("Using CLAUDE_3_5_HAIKU_ANTHROPIC model")
        from .anthropic_handler import AnthropicHandler
        return AnthropicHandler(model="CLAUDE_3_5_HAIKU_ANTHROPIC")
    elif name == "CLAUDE_3_7_AMAZON":
        print("Using CLAUDE_3_7_AMAZON model")
        from .bedrock_handler import BedrockHandler
        return BedrockHandler(model_type=ModelType.CLAUDE_3_7_AMAZON)
    elif name == "LITELLM":
        print("Using LiteLLM router")
        from .litellm_handler import LiteLLMHandler
        return LiteLLMHandler()
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 
