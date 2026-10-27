from enum import Enum


class ModelType(Enum):
    """
    Model identifiers used by the original bench runner config files.
    These are mapped to therapygym's in-tree LLM handlers in `bench_runner/llm_api.py`.
    """

    CLAUDE_3_7 = "claude_3_7"
    PHI_3_5 = "phi_3_5"
    GEMINI_2_0 = "gemini_2_0"
    O3_MINI = "o3_mini"
    O1 = "o1"
    LLAMA_4_MAVERICK = "llama_4_maverick"
    LLAMA_4_SCOUT = "llama_4_scout"
    DEEPSEEK = "deepseek"
    QWEN3_4B_LOCAL = "qwen3_4b_local"
    HUMAN = "human"


