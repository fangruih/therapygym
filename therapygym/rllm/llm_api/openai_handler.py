import os
import logging
import time
from typing import List, Dict, Optional
import openai
from threading import Lock
import httpx

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, tokens_per_minute: int = 200000):
        self.tokens_per_minute = tokens_per_minute
        self.used_tokens = 0
        self.window_start = time.time()
        self.lock = Lock()
        
    def wait_if_needed(self, requested_tokens: int):
        with self.lock:
            current_time = time.time()
            window_elapsed = current_time - self.window_start
            
            # Reset window if a minute has passed
            if window_elapsed >= 60:
                self.used_tokens = 0
                self.window_start = current_time
                window_elapsed = 0
            
            # Check if adding these tokens would exceed the limit
            if self.used_tokens + requested_tokens > self.tokens_per_minute:
                # Calculate wait time needed
                wait_time = 60 - window_elapsed
                if wait_time > 0:
                    logger.warning(f"Rate limit approaching, sleeping for {wait_time:.2f}s")
                    time.sleep(wait_time)
                    self.used_tokens = requested_tokens
                    self.window_start = time.time()
            else:
                self.used_tokens += requested_tokens

class OpenAIHandler:
    def __init__(
        self,
        model_type: Optional[object] = None,
        model_name: str = "o3-mini",
        api_key_env_vars: Optional[List[str]] = None,
        max_retries: int = 20,
        min_wait_time: float = 5.0,  # Minimum wait time between retries in seconds
        timeout: float = 600.0  # 10 minute timeout
    ):
        """Initialize OpenAI handler with rate limiting."""
        self.model_name = model_name
        self.max_retries = max_retries
        self.min_wait_time = min_wait_time
        self.timeout = timeout
        
        # Initialize rate limiter for o3-mini (200K TPM limit)
        self.rate_limiter = RateLimiter(tokens_per_minute=200000)
        
        # Get API key from environment
        api_key = os.getenv("openai_api_key")
        if not api_key:
            raise EnvironmentError(
                "openai_api_key environment variable not found"
            )
        
        # Initialize client with timeout
        self.client = openai.OpenAI(
            api_key=api_key,
            http_client=httpx.Client(timeout=timeout)
        )
        logger.info(f"Initialized OpenAI handler with model: {model_name}")

    def estimate_tokens(self, messages: List[Dict[str, str]], system_prompt: str = "") -> int:
        """Rough estimate of token count for rate limiting purposes."""
        total_chars = sum(len(msg["content"]) for msg in messages)
        if system_prompt:
            total_chars += len(system_prompt)
        # Rough estimate: 4 characters per token
        return total_chars // 4 + 100  # Add padding for safety

    def call_llm(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call OpenAI API with rate limiting and retries."""
        # Estimate tokens for rate limiting
        estimated_tokens = self.estimate_tokens(messages, system_prompt)
        
        # Convert messages to OpenAI format
        openai_messages = []
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        for msg in messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            openai_messages.append({
                "role": role,
                "content": msg["content"]
            })

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed(estimated_tokens)
                
                # Make API call with max_completion_tokens for o3-mini
                api_params = {
                    "model": self.model_name,
                    "messages": openai_messages,
                }
                
                # Use max_completion_tokens for o3-mini
                if self.model_name == "o3-mini":
                    api_params["max_completion_tokens"] = max_tokens if max_tokens else 2048
                else:
                    api_params["max_tokens"] = max_tokens if max_tokens else 2048
                
                response = self.client.chat.completions.create(**api_params)
                
                # Update used tokens with actual usage if available
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    self.rate_limiter.used_tokens += response.usage.total_tokens
                
                return response.choices[0].message.content
                
            except Exception as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                last_error = e
                
                if "rate_limit" in str(e).lower():
                    # For rate limit errors, wait longer
                    sleep_time = max(2.0, self.min_wait_time * (2 ** attempt))
                    logger.info(f"Rate limit hit, waiting {sleep_time} seconds...")
                    time.sleep(sleep_time)
                elif attempt < self.max_retries - 1:
                    # For other errors, use standard backoff
                    sleep_time = self.min_wait_time * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                continue

        # If we get here, all retries failed
        raise last_error or Exception("Failed to call OpenAI API after all retries")
