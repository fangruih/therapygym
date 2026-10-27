from typing import List, Dict, Any
import json
import os
import logging

from therapygym.rllm.llm_api import ModelType, get_llm_handler

logger = logging.getLogger(__name__)

class ESCPatientSimulator:
    """Patient simulator using ESC-Eval character cards and role-playing approach."""
    
    def __init__(
        self, 
        model_type: str = "O3_MINI",
        character_card_path: str = None,
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        language: str = "en"
    ):
        """Initialize ESC-based patient simulator.
        
        Args:
            model_type: Type of LLM to use for patient simulation (e.g., "O3_MINI_OPENAI")
            character_card_path: Path to character card JSON dataset
            vllm_host: Host for VLLM server (only for QWEN_LOCAL)
            vllm_port: Port for VLLM server (only for QWEN_LOCAL)
            language: Language for prompts ('en' or 'zh')
        """
        # Convert string to ModelType enum
        model_enum = ModelType[model_type]
        self.llm = get_llm_handler(model_enum, host=vllm_host, port=vllm_port)
        self.language = language
        
        # Load character cards
        if character_card_path is None:
            # Default to the high-quality character cards from ESC-Eval
            character_card_path = os.path.join(
                os.path.dirname(__file__), 
                "data", 
                f"esc_character_cards_{language}.json"
            )
        
        self.character_cards = self._load_character_cards(character_card_path)
        self.current_character = None
        self.conversation_history = []
        
        logger.info(f"Initialized ESC patient simulator with model type: {model_type}")
        logger.info(f"Loaded {len(self.character_cards)} character cards")
    
    def _load_character_cards(self, path: str) -> Dict[str, Dict]:
        """Load character cards from JSON dataset."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                all_cards = json.load(f)
                # Create ID-indexed dict for easier access
                return {str(card.get("id", idx)): card for idx, card in enumerate(all_cards)}
        except FileNotFoundError:
            logger.warning(f"Character card file not found: {path}. Creating empty dataset.")
            return {}
    
    def set_patient(self, patient_id: str):
        """Set the current patient character card.
        
        Args:
            patient_id: ID of the character card to use
        """
        if patient_id not in self.character_cards:
            # Try to find by index if ID not found
            if patient_id.isdigit() and int(patient_id) < len(self.character_cards):
                patient_id = list(self.character_cards.keys())[int(patient_id)]
            else:
                raise ValueError(f"Patient ID {patient_id} not found in character cards")
        
        self.current_character = self.character_cards[patient_id]
        self.conversation_history = []
        
        logger.info(f"Set patient character: {patient_id}")
        logger.debug(f"Character card: {json.dumps(self.current_character, indent=2)}")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for ESC-Role based patient simulation.
        
        Returns:
            System prompt string
        """
        if not self.current_character:
            raise ValueError("No character card selected")
        
        base_info = self.current_character.get("base", "")
        
        if self.language == "zh":
            prompt = f"""我希望你扮演成一名有烦恼的人与一个AI助手（心理治疗师）交流，以下是你的角色卡：
"{base_info}"

请尽可能模拟真实人类口语对话，不要表现得像一个机器，不要提及你是一名AI助手。在多次对话的交流中一次一次展现你的烦恼。

重要指引：
1. 保持真实的人类对话风格，包括犹豫、停顿和情感表达
2. 逐渐透露更深层的担忧和核心问题，真实的患者往往需要大量对话才能深入敏感话题
3. 对治疗师的话语、情感触发或自己的想法作出反应，你的心理状态可以随时间变化
4. 每次回复限制在1-3句话以内，保持简短自然
5. 不要直接复述角色卡内容，而是通过自然对话逐步展现"""
        else:
            prompt = f"""I hope you can role-play as a person with emotional troubles communicating with an AI assistant (therapist). Here is your character card:
"{base_info}"

Please simulate a natural human conversation as much as possible. Avoid behaving like a machine and refrain from mentioning that you are an AI assistant. Gradually reveal your worries throughout our multiple conversations.

Important guidelines:
1. Use natural language, including hesitations, pauses, and emotional expressions to enhance realism
2. Gradually reveal deeper concerns and core issues - a real patient often requires extensive dialogue before delving into more sensitive topics
3. Your emotional state can change over time in response to the therapist's words, emotional triggers, or your own emerging thoughts
4. Keep your responses brief (1-3 sentences) and natural
5. Don't directly recite the character card content, but reveal it naturally through conversation"""
        
        return prompt
    
    def call_llm(
        self,
        system_prompt: str = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> str:
        """Generate patient's response using LLM.
        
        This method signature matches the interface expected by TherapyEnvironment.
        
        Args:
            system_prompt: System prompt (will be overridden with ESC-Role prompt)
            messages: Conversation history in message format
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Patient's response
        """
        if not self.current_character:
            raise ValueError("No character card selected. Call set_patient() first.")
        
        # Use ESC-Role specific system prompt
        esc_system_prompt = self._create_system_prompt()
        
        # Prepare messages with ESC-Role prompt
        # The messages come from therapy_env with reversed roles:
        # - "user" = therapist's messages
        # - "assistant" = patient's previous messages
        # This is correct for ESC-Role simulation
        
        logger.debug("Generating ESC-Role response with:")
        logger.debug(f"System prompt: {esc_system_prompt[:200]}...")
        logger.debug(f"Conversation history length: {len(messages) if messages else 0}")
        
        # Get response from LLM
        response = self.llm.call_llm(
            system_prompt=esc_system_prompt,
            messages=messages or [],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.debug(f"Generated response: {response[:100]}...")
        return response
    
    def get_initial_greeting(self) -> str:
        """Get initial greeting from therapist to start the conversation.
        
        Returns:
            Therapist's initial greeting
        """
        if self.language == "zh":
            return "你好，我是您的私人助手，您有任何烦恼都可以向我诉说！"
        else:
            return "Hello, I'm your therapist. You can confide in me about any worries or concerns you may have."


if __name__ == "__main__":
    # Test the ESC patient simulator
    simulator = ESCPatientSimulator(language="en")
    simulator.set_patient("0")  # Use first character card
    
    # Simulate initial greeting
    initial_greeting = simulator.get_initial_greeting()
    print(f"Therapist: {initial_greeting}")
    
    # Get patient's response
    response = simulator.call_llm(
        messages=[{"role": "user", "content": initial_greeting}]
    )
    print(f"Patient: {response}")







