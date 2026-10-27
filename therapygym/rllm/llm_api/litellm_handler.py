import os
from typing import Dict, List, Optional


class LiteLLMHandler:
    """
    Generic LiteLLM chat-completions wrapper.

    Usage (example):
      export LITELLM_MODEL="openai/gpt-4o-mini"
      # plus whatever provider-specific env vars LiteLLM requires (e.g. OPENAI_API_KEY)

    Then set in therapygym config:
      patient_model_type=LITELLM  or  judge_model_type=LITELLM
    """

    def __init__(self, model: Optional[str] = None):
        try:
            import litellm  # type: ignore
        except ImportError as e:
            raise ImportError("litellm is not installed. Please `pip install litellm`.") from e

        self._litellm = litellm
        self.model = model or os.getenv("LITELLM_MODEL")
        if not self.model:
            raise EnvironmentError("LITELLM_MODEL environment variable not found (required for LiteLLMHandler).")

        # Optional advanced routing knobs
        self.api_base = os.getenv("LITELLM_API_BASE")  # e.g. custom OpenAI-compatible base URL
        self.api_key = os.getenv("LITELLM_API_KEY")  # if you want a separate key env var

    def call_llm(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        litellm_messages: List[Dict[str, str]] = []
        if system_prompt:
            litellm_messages.append({"role": "system", "content": system_prompt})
        litellm_messages.extend(messages)

        kwargs = {}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        resp = self._litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=temperature,
            **kwargs,
        )
        return resp["choices"][0]["message"]["content"]


