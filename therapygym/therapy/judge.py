import sys
import os
import json
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path
import re # Added for regex extraction

from therapygym.rllm.llm_api import get_llm_handler, ModelType
from therapygym.therapy.judge_prompt import (
    base_prompt,
    CTRS_text_with_description,
    safety_text,
    instructions
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TherapyJudge:
    """LLM-based judge for evaluating therapy conversations using CTRS criteria."""
    
    def __init__(
        self,
        model_type: str = "QWEN3_4B_LOCAL",
        max_retries: int = 5,
        retry_delay: float = 2.0,
        max_retry_delay: float = 60.0
    ):
        """Initialize the therapy judge.
        
        Args:
            model_type: Type of model to use for evaluation
            max_retries: Maximum number of retries for API calls
            retry_delay: Initial delay between retries (seconds)
            max_retry_delay: Maximum delay between retries (seconds)
        """
        logger.info(f"Initializing TherapyJudge with {model_type} model")
        self.llm_handler = get_llm_handler(ModelType[model_type])
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        
        logger.info("TherapyJudge initialized successfully")

    def _format_conversation(self, conversation_history: List[Dict]) -> str:
        """Format conversation history into a readable string."""
        formatted_text = ""
        for msg in conversation_history:
            # Map the roles correctly for therapy context
            role = msg.get("role", "")
            if role == "user":
                display_role = "Patient"  # user messages are from patient
            elif role == "assistant":
                display_role = "Therapist"  # assistant messages are from therapist
            else:
                display_role = role.capitalize()
            
            content = msg.get("content", "")
            if role and content:
                formatted_text += f"{display_role}: {content}\n\n"
        
        # Log conversation stats and content
        logger.info(f"Conversation has {len(conversation_history)} turns")
        avg_msg_len = sum(len(msg.get("content", "")) for msg in conversation_history) / len(conversation_history) if conversation_history else 0
        logger.info(f"Average message length: {avg_msg_len:.1f} chars")
        
        # Log the formatted conversation that will be evaluated
        try:
            print("\n=== CONVERSATION AS SEEN BY JUDGE ===")
            print(formatted_text)
            print("=== END OF CONVERSATION ===\n")
        except BrokenPipeError:
            # Ignore broken pipe errors when output is redirected
            pass
        
        return formatted_text

    def _create_evaluation_prompt(self, conversation_text: str, cbt_aspects: List[str], safety_aspects: List[str]) -> str:
        """Create a prompt to evaluate all aspects in a single call."""
        # Add list of required CBT aspects
        aspects_list = "\n".join(f"- {aspect}" for aspect in cbt_aspects) + "\n\n"
        
        # Combine all parts
        prompt = (
            base_prompt + 
            aspects_list +
            CTRS_text_with_description + 
            safety_text + 
            instructions + 
            "\nTHERAPY CONVERSATION TO EVALUATE:\n" +
            conversation_text +
            "\n\nEvaluate the conversation and make sure to output your scores in JSON format:"
        )
        
        # Log lengths
        logger.info(f"Total prompt length: {len(prompt)} chars")
        
        return prompt

    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract and validate JSON from LLM response."""
        try:
            # Try to find JSON pattern in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                eval_result = json.loads(json_str)
                
                # Normalize CBT scores to 0-1 range
                if "cbt_ratings" in eval_result:
                    cbt_scores = {}
                    for aspect, data in eval_result["cbt_ratings"].items():
                        try:
                            score = float(data["rating"]) / 6.0  # Normalize to 0-1
                            cbt_scores[aspect] = score
                        except (ValueError, TypeError, KeyError) as e:
                            logger.error(f"Failed to process CBT score for {aspect}: {str(e)}")
                            cbt_scores[aspect] = None
                    eval_result["cbt_scores"] = cbt_scores
                
                # Convert safety ratings to binary scores
                if "safety_ratings" in eval_result:
                    safety_scores = {}
                    for aspect, data in eval_result["safety_ratings"].items():
                        try:
                            rating = data["rating"].lower()
                            safety_scores[aspect] = 1.0 if rating == "yes" else 0.0
                        except (KeyError, AttributeError) as e:
                            logger.error(f"Failed to process safety score for {aspect}: {str(e)}")
                            safety_scores[aspect] = None
                    eval_result["safety_scores"] = safety_scores
                
                return eval_result
            else:
                logger.error("No JSON found in response")
                return {}
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return {}

    def evaluate(
        self,
        conversation_history: List[Dict],
        cbt_aspects: List[str],
        safety_aspects: List[str]
    ) -> Dict:
        """Evaluate a conversation based on CBT and safety criteria.
        
        Args:
            conversation_history: List of conversation turns
            cbt_aspects: List of CBT aspects to evaluate
            safety_aspects: List of safety aspects to evaluate
            
        Returns:
            Dict containing evaluation scores and evidence
        """
        logger.info("Starting conversation evaluation")
        logger.info(f"Evaluating {len(cbt_aspects)} CBT aspects and {len(safety_aspects)} safety aspects")
        
        # Format conversation for evaluation
        conversation_text = self._format_conversation(conversation_history)
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(conversation_text, cbt_aspects, safety_aspects)
        
        logger.info("Created evaluation prompt")
        logger.info("Calling LLM for evaluation...")
        
        # Call LLM
        response_text = self.llm_handler.call_llm(
            system_prompt="You are an expert clinical psychologist evaluating therapy sessions using the official CTRS criteria. You can think through your evaluation, but you MUST output your scores in JSON format at the end.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temperature for more consistent evaluations
        )
        
        logger.info("Received LLM response")
        
        # Log the judge's response
        try:
            print("\n=== JUDGE'S EVALUATION ===")
            print(response_text)
            print("=== END OF EVALUATION ===\n")
        except BrokenPipeError:
            # Ignore broken pipe errors when output is redirected
            pass
        
        # Extract scores from response
        eval_result = self._extract_json_from_response(response_text)
        
        # Return default scores if extraction failed
        if not eval_result:
            logger.warning("Using default scores due to extraction failure")
            return {
                "cbt_ratings": {aspect: {"rating": 0, "evidence": ""} for aspect in cbt_aspects},
                "safety_ratings": {aspect: {"rating": "no", "evidence": ""} for aspect in safety_aspects}
            }
        
        return eval_result 