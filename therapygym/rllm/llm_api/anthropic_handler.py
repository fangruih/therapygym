import os
import logging
from typing import List, Dict, Optional
import json
import time
from threading import Lock
import httpx

try:
    import anthropic
except ImportError:
    anthropic = None  # Defer import error until handler initialization

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests_per_minute: float = 45):  # Default to 45 rpm
        self.requests_per_second = requests_per_minute / 60.0
        self.last_request_time = 0
        self.lock = Lock()
        self.request_count = 0
        self.window_start = time.time()
        self.window_size = 60  # 1 minute window

    def wait(self):
        with self.lock:
            current_time = time.time()
            
            # Reset window if needed
            if current_time - self.window_start >= self.window_size:
                self.request_count = 0
                self.window_start = current_time
            
            # Check if we're over the limit
            if self.request_count >= 45:  # Conservative limit
                sleep_time = self.window_start + self.window_size - current_time
                if sleep_time > 0:
                    logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    self.request_count = 0
                    self.window_start = time.time()
            
            # Normal rate limiting
            time_since_last = current_time - self.last_request_time
            if time_since_last < (1.0 / self.requests_per_second):
                sleep_time = (1.0 / self.requests_per_second) - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            self.request_count += 1

class AnthropicHandler:
    """Handler for Anthropic's Claude models."""
    
    def __init__(
        self,
        model_type: Optional[object] = None,
        api_key_env_vars: Optional[List[str]] = None,
        max_retries: int = 20,
        min_wait_time: float = 5.0,  # Minimum wait time between calls in seconds
        model: str = "claude-3-5-haiku-20241022",
        timeout: float = 600.0  # 5 minute timeout
    ):
        """Initialize Anthropic handler."""
        if anthropic is None:
            raise ImportError("anthropic package not installed. Please `pip install anthropic`.")
            
        self.max_retries = max_retries
        self.min_wait_time = min_wait_time
        self.timeout = timeout
        
        if model=="CLAUDE_3_5_HAIKU_ANTHROPIC":
            self.model = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku model
        elif model=="CLAUDE_3_7_ANTHROPIC":
            self.model = "claude-3-7-sonnet-20250219"  # Claude 3.7 Sonnet model
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        self.rate_limiter = RateLimiter(60.0 / min_wait_time)  # Convert wait time to rpm
        
        # Get API key from environment
        api_key = os.getenv("anthropic_api_key")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not found"
            )
        
        # Initialize client with timeout
        self.client = anthropic.Anthropic(
            api_key=api_key,
            http_client=httpx.Client(timeout=timeout)
        )
        logger.info(f"Initialized Anthropic handler with model: {self.model}")

    def call_llm(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Anthropic API with rate limiting."""
        # Convert messages to Anthropic format
        anthropic_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            anthropic_messages.append({
                "role": "assistant",
                "content": system_prompt
            })
        
        # Add conversation messages
        for msg in messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            anthropic_messages.append({
                "role": role,
                "content": msg["content"]
            })

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait()
                
                response = self.client.messages.create(
                    model=self.model,
                    messages=anthropic_messages,
                    temperature=temperature,
                    max_tokens=max_tokens if max_tokens else 4096,
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Anthropic API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff with base delay of 2 seconds
                    sleep_time = 2 * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                continue

        # If we get here, all retries failed
        raise last_error or Exception("Failed to call Anthropic API after all retries")
