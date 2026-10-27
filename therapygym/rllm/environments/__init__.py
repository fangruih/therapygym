"""
Environments package.

Keep this module light: some environments depend on optional tool/reward stacks.
Core therapy PPO imports only `BaseEnv` via `therapygym.rllm.environments.base.base_env`.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "BaseEnv",
    "MultiTurnEnvironment",
    "SingleTurnEnvironment",
    "ToolEnvironment",
]

_LAZY = {
    "BaseEnv": "therapygym.rllm.environments.base.base_env:BaseEnv",
    "MultiTurnEnvironment": "therapygym.rllm.environments.base.multi_turn_env:MultiTurnEnvironment",
    # SingleTurnEnvironment pulls rewards; keep lazy so importing base_env doesn't import rewards.
    "SingleTurnEnvironment": "therapygym.rllm.environments.base.single_turn_env:SingleTurnEnvironment",
    "ToolEnvironment": "therapygym.rllm.environments.tools.tool_env:ToolEnvironment",
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
