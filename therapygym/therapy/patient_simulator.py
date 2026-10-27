from typing import List, Dict, Any
import json
import os
import logging

from therapygym.rllm.llm_api import ModelType, get_llm_handler
from therapygym.therapy.patient_prompting import build_patient_system_prompt, V7_PATIENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class PatientSimulator:
    """Patient simulator that uses LLM to generate patient responses."""
    
    def __init__(
        self, 
        model_type: str = "O3_MINI",
        vllm_host: str = "localhost",
        vllm_port: int = 8000
    ):
        """Initialize patient simulator.
        
        Args:
            model_type: Type of LLM to use for patient simulation
            vllm_host: Host for VLLM server (only for QWEN_LOCAL)
            vllm_port: Port for VLLM server (only for QWEN_LOCAL)
        """
        # Convert string to ModelType enum
        model_enum = ModelType[model_type]
        self.llm = get_llm_handler(model_enum, host=vllm_host, port=vllm_port)
        self.patient_profiles = self._load_patient_profiles()
        self.current_profile = None
        self.conversation_history = []
        
        self.patient_prompt = V7_PATIENT_SYSTEM_PROMPT
        
        logger.info(f"Initialized patient simulator with model type: {model_type}")
    
    def _load_patient_profiles(self) -> Dict[str, Dict]:
        """Load patient profiles from JSON dataset."""
        dataset_path = os.path.join(os.path.dirname(__file__), "data", "patient_psi_cm_dataset.json")
        with open(dataset_path, 'r') as f:
            all_patients = json.load(f)
            return {p["id"]: p for p in all_patients}
    
    def set_patient(self, patient_id: str):
        """Set the current patient profile.
        
        Args:
            patient_id: ID of the patient profile to use
        """
        if patient_id not in self.patient_profiles:
            raise ValueError(f"Patient ID {patient_id} not found in profiles")
        self.current_profile = self.patient_profiles[patient_id]
        self.conversation_history = []
        
        logger.info(f"Set patient profile: {patient_id}")
        logger.debug(f"Patient profile: {json.dumps(self.current_profile, indent=2)}")
    
    def generate_response(self, conversation_history: List[tuple[str, str]]) -> str:
        """Generate patient's response based on conversation history.
        
        Args:
            conversation_history: List of (speaker, text) tuples
            
        Returns:
            Patient's response
        """
        if not self.current_profile:
            raise ValueError("No patient profile selected")
            
        # Format conversation for LLM
        formatted_convo = self._format_conversation(conversation_history)
        
        # Add patient prompt to conversation history
        formatted_convo.insert(0, {
            "role": "system",
            "content": self.patient_prompt
        })
        
        # Format patient profile into proper prompt
        system_prompt = build_patient_system_prompt(self.current_profile)
        
        logger.debug("Generating response with:")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"Conversation: {json.dumps(formatted_convo, indent=2)}")
        
        # Get response from LLM
        response = self.llm.call_llm(
            system_prompt=system_prompt,
            messages=formatted_convo,
            temperature=0.8,
            max_tokens=300
        )
        
        logger.debug(f"Generated response: {response}")
        return response
    
    def _format_conversation(self, conversation_history: List[tuple[str, str]]) -> List[Dict[str, str]]:
        """Format conversation history for LLM.
        
        Args:
            conversation_history: List of (speaker, text) tuples
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        formatted = []
        for speaker, text in conversation_history:
            role = "assistant" if speaker == "patient" else "user"
            formatted.append({"role": role, "content": text})
        return formatted


if __name__ == "__main__":
    # Test the patient simulator
    simulator = PatientSimulator()
    simulator.set_patient("5-1")  # Use Chris's profile
    
    # Test with a sample conversation
    conversation = [
        ("patient", "I've been feeling really down lately, can't seem to get motivated to do anything."),
        ("therapist", "I hear that you're feeling down. Can you tell me more about when you first noticed these feelings?")
    ]
    
    response = simulator.generate_response(conversation)
    print("\n=== Final Response ===")
    print(response) 