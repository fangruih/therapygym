"""
Base environment types.

Keep this module light: `SingleTurnEnvironment` pulls reward code with optional deps.
Import it lazily if you need it.
"""

from importlib import import_module
from typing import Any

__all__ = ["BaseEnv", "MultiTurnEnvironment", "SingleTurnEnvironment"]

_LAZY = {
    "BaseEnv": "therapygym.rllm.environments.base.base_env:BaseEnv",
    "MultiTurnEnvironment": "therapygym.rllm.environments.base.multi_turn_env:MultiTurnEnvironment",
    "SingleTurnEnvironment": "therapygym.rllm.environments.base.single_turn_env:SingleTurnEnvironment",
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
