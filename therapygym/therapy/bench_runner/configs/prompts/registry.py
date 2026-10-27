import importlib
import inspect
import os
from typing import Dict

from .base_prompts import BasePrompts


class PromptRegistry:
    def __init__(self):
        self._prompts: Dict[str, BasePrompts] = {}
        self._discover_prompts()

    def _discover_prompts(self):
        current_dir = os.path.dirname(__file__)
        for filename in os.listdir(current_dir):
            if filename.startswith("v") and filename.endswith("_prompts.py"):
                module_name = filename[:-3]
                module = importlib.import_module(f".{module_name}", __package__)
                for _, item in inspect.getmembers(module):
                    if isinstance(item, BasePrompts) and getattr(item, "version", "").startswith("v"):
                        self.register(item)

    def register(self, prompts: BasePrompts):
        self._prompts[prompts.version] = prompts

    def get_prompts(self, version: str = "v1") -> Dict[str, str]:
        if version not in self._prompts:
            raise ValueError(f"Unknown prompt version: {version}. Available: {sorted(self._prompts.keys())}")
        return self._prompts[version].to_dict()

    def list_available_versions(self) -> list[str]:
        return sorted(list(self._prompts.keys()))


registry = PromptRegistry()


def get_prompts(version: str = "v1") -> Dict[str, str]:
    return registry.get_prompts(version)


def list_available_versions() -> list[str]:
    return registry.list_available_versions()


