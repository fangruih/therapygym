import requests
from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

class VLLMHandler:
    """Handler for local VLLM-served Qwen model."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, model: str = "Qwen/Qwen3-4B"):
        """Initialize VLLM handler.
        
        Args:
            host: VLLM server host
            port: VLLM server port
        """
        self.endpoint = f"http://{host}:{port}/v1/chat/completions"
        self.model = model
        logger.info(f"Initialized VLLM handler with endpoint: {self.endpoint}")
    
    def call_llm(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000  # Increased from 1000 to 2000
    ) -> str:
        """Call local VLLM server.
        
        Args:
            system_prompt: System prompt for the model
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Add system prompt if not already in messages
        if not messages or messages[0].get('role') != 'system':
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Prepare request
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        logger.debug(f"Sending request to VLLM: {json.dumps(request, indent=2)}")
        
        try:
            # Call VLLM server
            response = requests.post(
                self.endpoint,
                json=request,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"VLLM request failed: {response.text}")
                raise Exception(f"VLLM request failed: {response.text}")
                
            result = response.json()
            logger.info("=== RAW VLLM RESPONSE (START) ===")
            logger.info(json.dumps(result, indent=2))
            logger.info("=== RAW VLLM RESPONSE (END) ===")
            response_text = result['choices'][0]['message']['content']
            
            # Process response - remove thinking process if present
            if "<think>" in response_text and "</think>" in response_text:
                processed_text = response_text.split("</think>")[-1].strip()
                logger.debug(f"Processed response (removed thinking): {processed_text}")
                return processed_text
            
            return response_text
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to VLLM server at {self.endpoint}")
            raise
        except Exception as e:
            logger.error(f"Error calling VLLM: {str(e)}")
            raise 
