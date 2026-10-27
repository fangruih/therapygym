"""
Agents package.

Important: keep this module light on import side-effects.
Some agents pull optional heavyweight dependencies (web tools, code sandboxes, etc.).
Those should be imported lazily so core training/inference paths don't require them.
"""

from importlib import import_module
from typing import Any

__all__ = [
    # Common agents (resolved lazily)
    "ToolAgent",
    "MathAgent",
    "MiniWobAgent",
    "FrozenLakeAgent",
    "CompetitionCodingAgent",
    "WebArenaAgent",
]

_LAZY = {
    "ToolAgent": "therapygym.rllm.agents.tool_agent:ToolAgent",
    "MathAgent": "therapygym.rllm.agents.math_agent:MathAgent",
    "MiniWobAgent": "therapygym.rllm.agents.miniwob_agent:MiniWobAgent",
    "FrozenLakeAgent": "therapygym.rllm.agents.frozenlake_agent:FrozenLakeAgent",
    "CompetitionCodingAgent": "therapygym.rllm.agents.code_agent:CompetitionCodingAgent",
    "WebArenaAgent": "therapygym.rllm.agents.webarena_agent:WebArenaAgent",
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if not target:
        raise AttributeError(name)
    mod_path, attr = target.split(":")
    mod = import_module(mod_path)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
