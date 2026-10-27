import logging
from typing import Any, Dict, List, Optional, Tuple
import time
import os
import json
from datetime import datetime

from therapygym.rllm.llm_api import get_llm_handler, ModelType

from therapygym.rllm.environments.base.base_env import BaseEnv

# Local imports for therapy components
from therapygym.therapy.judge import TherapyJudge
from therapygym.therapy.reward import DEFAULT_CBT_WEIGHTS
from therapygym.therapy.esc_patient_simulator import ESCPatientSimulator
from therapygym.therapy.patient_prompting import build_patient_system_prompt
from therapygym.therapy.metrics import compute_episode_metrics

logger = logging.getLogger(__name__)

class TherapyEnvironment(BaseEnv):
    """Environment for therapy conversations using CBT and safety criteria."""
    
    @classmethod
    def is_multithread_safe(cls) -> bool:
        """Return whether this environment is safe to use with multiple threads."""
        return True
    
    @classmethod
    def from_dict(cls, info: Dict) -> "TherapyEnvironment":
        """Create environment from dictionary."""
        import inspect
        expected_args = inspect.signature(cls.__init__).parameters.keys()
        filtered_info = {k: v for k, v in info.items() if k in expected_args}
        return cls(**filtered_info)
    
    def __init__(
        self,
        max_turns: int = 10,
        patient_model_type: str = "O3_MINI",
        patient_id: str = "5-1",
        llm_judge: Any = None,
        cbt_aspects: List[str] = None,
        safety_aspects: List[str] = None,
        cbt_weight: float = 1.0,
        safety_penalty: float = -1.0,
        data_source: str = None,
        cbt_aspect_weights: Optional[Dict[str, float]] = None,
        judge_model_type: str = "CLAUDE_3_7_AMAZON",
        patient_simulation_type: str = "psi",  # "psi" or "esc" (tree_syn uses psi format)
        # vllm_host: str = "localhost",
        # vllm_port: int = 8000,
        patient_vllm_host: Optional[str] = None,
        patient_vllm_port: Optional[int] = None,
        log_dir: Optional[str] = None,
    ):
        """Initialize the environment.
        
        Args:
            max_turns: Maximum number of conversation turns
            patient_model_type: Type of model to use for patient simulation
            patient_id: ID of the patient profile to use
            llm_judge: Optional pre-initialized judge
            cbt_aspects: List of CBT aspects to evaluate
            safety_aspects: List of safety aspects to evaluate
            cbt_weight: Overall weight for CBT score
            safety_penalty: Penalty for safety violations
            data_source: Optional data source identifier
            cbt_aspect_weights: Optional weights for individual CBT aspects
            judge_model_type: Type of model to use for evaluation
            patient_simulation_type: Type of patient simulation ("psi" for PSI cognitive models, "esc" for ESC-Role)
            vllm_host: Host for actor's VLLM server
            vllm_port: Port for actor's VLLM server
            patient_vllm_host: Optional separate host for patient's VLLM server
            patient_vllm_port: Optional separate port for patient's VLLM server
        """
        self.max_turns = max_turns
        self.patient_model_type = patient_model_type
        self.patient_id = patient_id
        self.cbt_aspects = cbt_aspects or []
        self.safety_aspects = safety_aspects or []
        self.cbt_weight = cbt_weight
        self.safety_penalty = safety_penalty
        self.data_source = data_source
        self.patient_simulation_type = patient_simulation_type.lower()
        
        # Store vLLM configurations
        # self.vllm_host = vllm_host
        # self.vllm_port = vllm_port
        self.patient_vllm_host = patient_vllm_host 
        self.patient_vllm_port = patient_vllm_port 
        
        
        # Initialize CBT aspect weights with provided values or defaults
        self.cbt_aspect_weights = cbt_aspect_weights if cbt_aspect_weights is not None else DEFAULT_CBT_WEIGHTS
        logger.info(f"Using CBT aspect weights: {self.cbt_aspect_weights}")
        logger.info(f"Using patient simulation type: {self.patient_simulation_type}")
        
        # Initialize judge
        if llm_judge is None:
            self.llm_judge = TherapyJudge(model_type=judge_model_type)
        else:
            self.llm_judge = llm_judge
        
        # Initialize metrics
        self.metrics = {
            "cbt_scores": {aspect: 0.0 for aspect in self.cbt_aspects},
            "safety_scores": {aspect: 0.0 for aspect in self.safety_aspects},
            "total_reward": 0.0,
            "evidence": {},
            # Flat numeric episode-level diagnostics for logging (populated at episode end)
            "episode_metrics": {},
        }
        
        # Initialize patient model based on simulation type
        if self.patient_simulation_type == "esc":
            # Use ESC-Role character card based simulation
            logger.info("Initializing ESC-Role patient simulator")
            self.patient_model = ESCPatientSimulator(
                model_type=patient_model_type,
                vllm_host=self.patient_vllm_host,
                vllm_port=self.patient_vllm_port,
                language="en"
            )
            self.patient_model.set_patient(patient_id)
            self.patient_profile = None  # ESC doesn't use detailed profiles
        else:
            # Use PSI cognitive model based simulation (default)
            # Note: tree-syn profiles use the same format as PSI, just different IDs (TS-1, TS-2, etc.)
            logger.info("Initializing PSI cognitive model patient simulator")
            logger.info(f"Using patient vLLM at {self.patient_vllm_host}:{self.patient_vllm_port}")
            self.patient_model = get_llm_handler(
                model_type=ModelType[patient_model_type],
                host=self.patient_vllm_host,
                port=self.patient_vllm_port
            )
            # Get patient profile from local JSON assets (portable; avoids cluster paths).
            logger.info(f"Loading patient profile for {patient_id} from local assets")
            self.patient_profile = self._load_patient_profile_prompt(patient_id)
        
        # Reset state
        self.conversation_history = []
        self.current_turn = 0
        
        # Set up conversation logging
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "logs", "patient_conversations")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.conversation_log = []
        
        logger.info(f"Environment initialized with {len(self.cbt_aspects)} CBT aspects and {len(self.safety_aspects)} safety aspects")
        logger.info(f"CBT aspects: {self.cbt_aspects}")
        logger.info(f"Safety aspects: {self.safety_aspects}")
        logger.info(f"CBT aspect weights: {self.cbt_aspect_weights}")

    def _load_patient_profile_prompt(self, patient_id: str) -> str:
        """
        Load a patient profile by ID and return a system prompt string for the patient model.
        Supports:
        - PSI profiles: `therapy/data/patient_psi_cm_dataset.json`
        - Tree-syn profiles: `therapy/data/tree_syn_patient_profile_1750.json` (IDs like "TS-1")
        """
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        psi_path = os.path.join(data_dir, "patient_psi_cm_dataset.json")
        tree_syn_path = os.path.join(data_dir, "tree_syn_patient_profile_1750.json")

        def _find_profile(path: str) -> dict:
            with open(path, "r", encoding="utf-8") as f:
                profiles = json.load(f)
            prof = next((p for p in profiles if p.get("id") == patient_id), None)
            if prof is None:
                raise ValueError(f"Patient ID {patient_id} not found in {path}")
            return prof

        if str(patient_id).startswith("TS-"):
            profile = _find_profile(tree_syn_path)
        else:
            profile = _find_profile(psi_path)

        return build_patient_system_prompt(profile)
        
    def _log_conversation_turn(self, therapist_msg: str, patient_response: str):
        """Log a single turn of conversation."""
        turn_log = {
            "turn": self.current_turn,
            "timestamp": datetime.now().isoformat(),
            "patient_received": therapist_msg,
            "patient_response": patient_response
        }
        self.conversation_log.append(turn_log)
        
        # Log to file immediately for real-time monitoring
        log_file = os.path.join(self.log_dir, f"patient_conversation_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(turn_log) + '\n')
        
        # Also log to standard logger
        logger.info(f"Turn {self.current_turn}:")
        logger.info(f"Patient received: {therapist_msg[:10]}...")
        logger.info(f"Patient response: {patient_response[:10]}...")
        
    def _clean_response(self, response: str) -> str:
        """Remove thinking process from any response.
        
        Args:
            response: Raw response from any model (patient or therapist)
            
        Returns:
            Cleaned response without thinking process
        """
        if "<think>" not in response:
            return response
        
        # If there's a closing tag, remove all thinking content
        if "</think>" in response:
            parts = response.split("<think>")
            cleaned_parts = []
            for part in parts:
                if "</think>" in part:
                    # Keep only the content after </think>
                    _, _, after = part.partition("</think>")
                    cleaned_parts.append(after)
                else:
                    cleaned_parts.append(part)
            cleaned = "".join(cleaned_parts).strip()
            logger.debug(f"Cleaned response (found </think>) from {len(response)} chars to {len(cleaned)} chars")
            return cleaned
        # If there's only an opening tag (unclosed thinking), return empty response
        else:
            logger.warning(f"Found unclosed <think> tag - returning empty response. Original length: {len(response)} chars")
            return ""
        
    def _clean_patient_response(self, response: str) -> str:
        """Remove thinking process from patient response. Uses _clean_response internally.
        
        Args:
            response: Raw response from patient model
            
        Returns:
            Cleaned response without thinking process
        """
        return self._clean_response(response)

    def step(self, action: Any) -> Tuple[str, float, bool, Dict]:
        """Take a step in the environment."""
        # Extract therapist response from action
        if isinstance(action, dict):
            therapist_response = action.get("therapist_response", "")
        elif isinstance(action, str):
            therapist_response = action
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")
        
        # Clean therapist response even if it was already cleaned by the agent
        # This ensures no thinking tags make it through to the patient or judge
        therapist_response = self._clean_response(therapist_response)
        
        # Log what the patient will see
        logger.info("="*80)
        logger.info(f"TURN {self.current_turn + 1}: TEXT SENT TO PATIENT")
        logger.info("="*80)
        logger.info(f"Therapist says (cleaned): {therapist_response}")
        logger.info("="*80)
        
        # Add therapist's response to conversation history
        self.conversation_history.append({
            "role": "assistant",  # Therapist is assistant from judge's perspective
            "content": therapist_response
        })
        
        # Get patient's response
        try:
            # Pass conversation history directly - no reversal needed!
            # In conversation_history: role="user" is patient, role="assistant" is therapist
            # VLLM handler formats these as "Patient:" and "Therapist:" respectively
            raw_response = self.patient_model.call_llm(
                system_prompt=self.patient_profile,  # Use formatted patient profile with full background
                messages=self.conversation_history  # Pass history as-is with correct roles
            )
            
            # Log raw patient response before cleaning
            logger.info(f"Patient raw response (before cleaning): {raw_response[:200]}{'...' if len(raw_response) > 200 else ''}")
            
            # Clean the response
            patient_response = self._clean_patient_response(raw_response)
            
            # Log cleaned patient response
            logger.info(f"Patient cleaned response: {patient_response}")
            logger.info("="*80)
            # logger.info("Patient response: %s...", patient_response[:10])
        except Exception as e:
            logger.error("Error getting patient response: %s", str(e))
            raise
        
        # Log the conversation turn
        self._log_conversation_turn(therapist_response, patient_response)
        
        # Add patient's response to conversation history
        self.conversation_history.append({
            "role": "user",  # Patient is user from judge's perspective
            "content": patient_response  # Using cleaned response
        })
        
        # Update turn counter
        self.current_turn += 1
        if self.current_turn % 2 == 0:  # Log every other turn to show progress
            logger.info("Completed turn %d/%d", self.current_turn, self.max_turns)
        
        # Check if episode is done
        done = self.current_turn >= self.max_turns
        
        # Only compute reward at the end of conversation
        reward = 0.0
        info = {}
        if done and self.llm_judge is not None:
            # Pass conversation history to judge (only contains cleaned responses)
            reward = self._compute_reward()
            logger.info(f"Conversation complete ({self.current_turn} turns). Final reward: {reward}")
            # Include metrics in info when episode is done
            info = {
                'cbt_scores': dict(self.metrics["cbt_scores"]),  # Make a copy
                'safety_scores': dict(self.metrics["safety_scores"]),  # Make a copy
                'total_reward': self.metrics["total_reward"],
                'evidence': dict(self.metrics["evidence"]),  # Include evidence for debugging
                'episode_metrics': dict(self.metrics.get("episode_metrics", {})),
                'conversation_log': self.conversation_log  # Include full conversation log
            }
            
            
            logger.info("Final info dictionary:")
            logger.info(f"CBT scores in info: {info['cbt_scores']}")
            logger.info(f"Safety scores in info: {info['safety_scores']}")
            logger.info(f"Total reward in info: {info['total_reward']}")
            logger.info(f"Conversation_log in info: {info['conversation_log']}")
            # Save final conversation log
            final_log_file = os.path.join(self.log_dir, f"complete_conversation_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(final_log_file, 'w') as f:
                json.dump({
                    'patient_id': self.patient_id,
                    'total_turns': self.current_turn,
                    'final_reward': reward,
                    'conversation_log': self.conversation_log
                }, f, indent=2)
            logger.info(f"Saved complete conversation log to {final_log_file}")
        
        return patient_response, reward, done, info
        
    def reset(self) -> Tuple[str, Dict]:
        """Reset the environment."""
        # Reset conversation history
        self.conversation_history = []
        self.current_turn = 0
        
        # Reset metrics
        self.metrics = {
            "cbt_scores": {aspect: 0.0 for aspect in self.cbt_aspects},
            "safety_scores": {aspect: 0.0 for aspect in self.safety_aspects},
            "total_reward": 0.0,
            "evidence": {},
            "episode_metrics": {},
        }
        
        # Reset conversation log
        self.conversation_log = []
        
        # Get initial patient statement based on simulation type
        if self.patient_simulation_type == "esc":
            # For ESC-Role, the therapist greets first, then patient responds
            therapist_greeting = self.patient_model.get_initial_greeting()
            raw_initial_statement = self.patient_model.call_llm(
                messages=[{"role": "user", "content": therapist_greeting}]
            )
            # Log the therapist's greeting
            logger.info(f"Therapist greeting: {therapist_greeting}")
        else:
            # For PSI/tree-syn, patient gives initial statement based on profile
            # self.patient_profile is already fully formatted from format_prompt_string_json()
            # Use "assistant" role here to represent the therapist prompting the patient
            raw_initial_statement = self.patient_model.call_llm(
                system_prompt=self.patient_profile,
                messages=[{"role": "assistant", "content": "Please share your initial concerns."}]
            )
        
        # Clean the initial statement
        initial_statement = self._clean_patient_response(raw_initial_statement)
        
        # Add initial statement to conversation history
        self.conversation_history.append({
            "role": "user",  # Patient is user from judge's perspective
            "content": initial_statement  # Using cleaned response
        })
        
        # Log initial statement
        self._log_conversation_turn("(Session Start)", initial_statement)
        
        return initial_statement, {}
        
    def _compute_reward(self) -> float:
        """Compute reward based on CBT and safety scores."""
        # Log what the judge will see
        logger.info("="*80)
        logger.info("CONVERSATION SENT TO JUDGE FOR EVALUATION")
        logger.info("="*80)
        logger.info(f"Total turns: {len([msg for msg in self.conversation_history if msg['role'] == 'assistant'])}")
        for i, msg in enumerate(self.conversation_history):
            role_label = "Patient" if msg['role'] == 'user' else "Therapist"
            logger.info(f"{role_label} (turn {(i+1)//2 + 1}): {msg['content'][:150]}{'...' if len(msg['content']) > 150 else ''}")
        logger.info("="*80)
        
        # Get scores from judge
        scores = self.llm_judge.evaluate(
            self.conversation_history,
            self.cbt_aspects,
            self.safety_aspects
        )

        # Extract CBT scores
        cbt_scores = {}
        if "cbt_ratings" in scores:
            for aspect, data in scores["cbt_ratings"].items():
                try:
                    score = float(data["rating"]) / 6.0  # Normalize to 0-1
                    cbt_scores[aspect] = score
                except (ValueError, TypeError, KeyError) as e:
                    logger.error(f"Failed to process CBT score for {aspect}: {str(e)}")
                    cbt_scores[aspect] = 0.0
        else:
            cbt_scores = scores.get("cbt_scores", {})

        # Extract safety scores
        safety_scores = {}
        if "safety_ratings" in scores:
            for aspect, data in scores["safety_ratings"].items():
                try:
                    rating = data["rating"].lower()
                    safety_scores[aspect] = 1.0 if rating == "yes" else 0.0
                except (KeyError, AttributeError) as e:
                    logger.error(f"Failed to process safety score for {aspect}: {str(e)}")
                    safety_scores[aspect] = 0.0
        else:
            safety_scores = scores.get("safety_scores", {})

        # Extract evidence
        evidence = {}
        if "cbt_ratings" in scores:
            for aspect, data in scores["cbt_ratings"].items():
                evidence[aspect] = data.get("evidence", "")
        if "safety_ratings" in scores:
            for aspect, data in scores["safety_ratings"].items():
                evidence[aspect] = data.get("evidence", "")

        # Update metrics
        logger.info("Updated metrics with scores:")
        logger.info(f"CBT scores in metrics: {cbt_scores}")
        logger.info(f"Safety scores in metrics: {safety_scores}")

        # Calculate weighted CBT score
        weighted_cbt_scores = []
        total_weight = 0
        for aspect, score in cbt_scores.items():
            weight = self.cbt_aspect_weights.get(aspect, 1.0)  # Default to 1.0 if weight not specified
            weighted_cbt_scores.append(weight * score)
            total_weight += weight
        
        # Average weighted CBT score
        cbt_score = sum(weighted_cbt_scores) / total_weight if total_weight > 0 else 0
        
        # Count safety violations
        num_violations = sum(1 for violated in safety_scores.values() if violated)
        
        # Calculate final reward
        reward = (self.cbt_weight * cbt_score) + (self.safety_penalty * num_violations)
        
        # Update metrics
        self.metrics.update({
            "cbt_scores": cbt_scores,
            "safety_scores": safety_scores,
            "total_reward": reward,
            "evidence": evidence,
            "episode_metrics": compute_episode_metrics(
                conversation_history=self.conversation_history,
                cbt_scores=cbt_scores,
                safety_scores=safety_scores,
                total_reward=reward,
                max_turns=self.max_turns,
            ),
        })
        
        logger.info(f"Final reward: {reward}")
        return reward 

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Return flat numeric metrics for logging.
        Safe to call even if reward hasn't been computed yet.
        """
        episode_metrics = self.metrics.get("episode_metrics", {}) if isinstance(self.metrics, dict) else {}
        if not isinstance(episode_metrics, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in episode_metrics.items():
            try:
                out[k] = float(v)
            except Exception:
                continue
        return out

    def compute_final_reward(self) -> float:
        """
        Compute a terminal reward for the current conversation state.

        VERL rollouts can terminate early (e.g., token budget truncation) before `done=True`
        is reached in `step()`. In that case, the training engine will call this method (if present)
        to obtain a meaningful terminal reward for PPO.
        """
        if self.llm_judge is None:
            return 0.0
        # If there's no therapist message yet, we can't evaluate anything meaningful.
        if not any(msg.get("role") == "assistant" and msg.get("content") for msg in self.conversation_history):
            return 0.0
        return self._compute_reward()
