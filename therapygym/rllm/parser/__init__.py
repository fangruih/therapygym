"""
Parser package.

Keep this module light: tool parsers may pull tool definitions which in turn can pull
optional dependencies. We load parsers lazily when requested.
"""

from importlib import import_module


def get_tool_parser(parser_name: str):
    if parser_name == "r1":
        return import_module("therapygym.rllm.parser.tool_parser.r1_tool_parser").R1ToolParser
    if parser_name == "qwen":
        return import_module("therapygym.rllm.parser.tool_parser.qwen_tool_parser").QwenToolParser
    raise KeyError(f"Unknown tool parser: {parser_name}")


__all__ = ["get_tool_parser"]
