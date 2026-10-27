import os
import logging
from typing import List, Dict, Optional
import json
import time
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class BedrockHandler:
    """Handler for Amazon Bedrock's Claude models."""
    
    def __init__(
        self,
        model_type: Optional[object] = None,
        api_key_env_vars: Optional[List[str]] = None,
        region: str = "us-east-1",
        max_retries: int = 3,
        min_wait_time: float = 2.0  # Minimum wait time between calls in seconds
    ):
        """Initialize Bedrock handler."""
        if boto3 is None:
            raise ImportError("boto3 package not installed. Please `pip install boto3`.")
            
        self.region = region
        self.max_retries = max_retries
        self.min_wait_time = min_wait_time
        self.model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"  # Claude 3.7 Sonnet
        self.last_call_time = 0
        
        # Initialize client
        self._initialize_client()
        logger.info(f"Initialized Bedrock handler with model: {self.model}")

    def _initialize_client(self):
        """Initialize or reinitialize the Bedrock client."""
        try:
            # Initialize client exactly like the reference code
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region
            )
            logger.info(f"Initialized Bedrock client in region {self.region}")
                
        except Exception as e:
            raise EnvironmentError(f"Failed to initialize Bedrock client: {str(e)}")

    def _wait_if_needed(self):
        """Ensure minimum wait time between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.min_wait_time:
            sleep_time = self.min_wait_time - time_since_last
            logger.debug(f"Waiting {sleep_time:.3f}s between API calls")
            time.sleep(sleep_time)
        self.last_call_time = time.time()

    def call_llm(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Bedrock API."""
        # Convert messages to Bedrock format
        bedrock_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            bedrock_messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Add conversation messages
        for msg in messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            bedrock_messages.append({
                "role": role,
                "content": [{"type": "text", "text": msg["content"]}]
            })

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Ensure minimum wait time between calls
                self._wait_if_needed()
                
                response = self.client.invoke_model(
                    modelId=self.model,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": bedrock_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens if max_tokens else 4096,
                    })
                )
                response_body = json.loads(response['body'].read())
                return response_body['content'][0]['text']
            except Exception as e:
                logger.error(f"Bedrock API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff with base delay of 2 seconds
                    sleep_time = 2 * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                continue

        # If we get here, all retries failed
        raise last_error or Exception("Failed to call Bedrock API after all retries") 