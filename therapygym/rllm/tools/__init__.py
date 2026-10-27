"""
Tools package.

Keep imports lazy: many tools require optional third-party services/dependencies.
Core PPO training for therapy does not need these tools to import.
"""

from importlib import import_module
from typing import Any

__all__ = ["ToolRegistry", "tool_registry", "get_default_tools"]


def get_default_tools() -> dict[str, Any]:
    """Return the default tool mapping (imports tool implementations lazily)."""
    from therapygym.rllm.tools.code_tools import PythonInterpreter
    from therapygym.rllm.tools.web_tools import FirecrawlTool, GoogleSearchTool, TavilyExtractTool, TavilySearchTool

    return {
        "python": PythonInterpreter,
        "google_search": GoogleSearchTool,
        "firecrawl": FirecrawlTool,
        "tavily-extract": TavilyExtractTool,
        "tavily-search": TavilySearchTool,
    }


def _init_registry():
    from therapygym.rllm.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_all(get_default_tools())
    return reg


# Lazy singleton registry
tool_registry = None


def __getattr__(name: str) -> Any:
    global tool_registry
    if name == "ToolRegistry":
        return import_module("therapygym.rllm.tools.registry").ToolRegistry
    if name == "tool_registry":
        if tool_registry is None:
            tool_registry = _init_registry()
        return tool_registry
    raise AttributeError(name)
