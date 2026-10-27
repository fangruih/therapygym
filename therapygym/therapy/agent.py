"""
This module will contain the therapy agent implementation.
It will be responsible for:
1. Generating appropriate therapy responses
2. Maintaining conversation context
3. Applying CBT techniques
4. Ensuring safety guidelines
"""
import copy 
from typing import Dict, List, Optional
from therapygym.rllm.agents.agent import BaseAgent, Action, Step, Trajectory
from dataclasses import asdict

class SimpleTherapyAgent(BaseAgent):
    """Specialized agent for therapy conversations.
    Inherits directly from BaseAgent to avoid tool call parsing.
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        accumulate_thinking: bool = False,
        max_tokens: int = 500,
        **kwargs
    ):
        """Initialize therapy agent.
        
        Args:
            system_prompt: System prompt for the agent (optional, uses default if not provided)
            system_prompt_file: Path to file containing system prompt (overrides system_prompt)
            accumulate_thinking: Whether to accumulate thinking tags
            max_tokens: Maximum tokens for therapist responses (default: 500)
            **kwargs: Additional arguments
        """
        
        # Load system prompt from file if provided, otherwise use provided prompt or default
        if system_prompt_file:
            with open(system_prompt_file, 'r') as f:
                self.system_prompt = f.read().strip()
        else:
            self.system_prompt = system_prompt or "You are a professional CBT therapist providing help online. Be colloquial and concise."
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking
        self.current_observation = None
        self.max_tokens = max_tokens

        self.reset()
    
    def _strip_thinking_tags(self, text: str) -> str:
        """Remove thinking tags and their content from the response.
        
        Args:
            text: Text that may contain thinking tags
            
        Returns:
            Text with thinking tags and their content removed
        """
        if "<think>" not in text:
            return text
        
        # If there's a closing tag, remove everything before and including it
        if "</think>" in text:
            parts = text.split("<think>")
            cleaned_parts = []
            for part in parts:
                if "</think>" in part:
                    # Keep only the content after </think>
                    _, _, after = part.partition("</think>")
                    cleaned_parts.append(after)
                else:
                    cleaned_parts.append(part)
            return "".join(cleaned_parts).strip()
        else:
            # If there's only an opening tag (unclosed thinking), return empty response
            return ""
    
    def _format_conversation_history(self, history: List[Dict]) -> List[Dict]:
        """Format conversation history for the model.
        Ensures proper role assignment and message formatting.
        """
        formatted_history = []
        for entry in history:
            if entry["role"] == "patient":
                # Patient messages become user messages
                formatted_history.append({
                    "role": "user",
                    "content": entry["content"]
                })
            else:
                # Therapist messages become assistant messages
                formatted_history.append({
                    "role": "assistant",
                    "content": entry["content"]
                })
        return formatted_history
    
    def _prepare_messages(self, observation: Dict, conversation_history: Optional[List[Dict]] = None) -> List[Dict]:
        """Prepare the messages for the model.
        Includes conversation history and current patient statement.
        """
        messages = []
        
        # Add formatted conversation history
        if conversation_history:
            messages.extend(self._format_conversation_history(conversation_history))
        
        # Add current patient statement
        if isinstance(observation, dict) and "patient_statement" in observation:
            messages.append({
                "role": "user",
                "content": observation["patient_statement"]
            })
        elif isinstance(observation, str):
            messages.append({
                "role": "user",
                "content": observation
            })
        
        return messages
    
    def get_action(self, observation: str) -> Dict:
        """
        This agent is designed to be used with `rllm.engine.AgentExecutionEngine` / VERL,
        where the trainer handles model generation and calls `update_from_model()`.
        """
        raise NotImplementedError(
            "SimpleTherapyAgent does not directly call an LLM. Use it with AgentExecutionEngine/VERL."
        )
    
    def update_from_env(self, observation: Dict, reward: float, done: bool, info: dict, **kwargs):
        """Updates the agent's state based on environment feedback."""
        self.current_observation = observation
        # Add new messages to existing ones
        new_messages = self._prepare_messages(observation, None)  # Don't use conversation_history since we maintain it ourselves
        self.messages.extend(new_messages)
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """Updates the agent's state based on the model's response."""
        # Store full response (with thinking) in internal messages and trajectory
        self.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Create new step with full response in trajectory
        new_step = Step(
            chat_completions=self.messages.copy(),  # Store full messages including thinking
            action=self._strip_thinking_tags(response),  # Clean response for environment
            model_response=response,  # Keep full response for reference
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)
        
        # Return cleaned response for environment
        return Action(action=self._strip_thinking_tags(response))
    
    def reset(self):
        """Resets the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None
    
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]