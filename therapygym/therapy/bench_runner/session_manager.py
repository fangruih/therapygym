import time
from typing import Dict, List, Optional, Union

from .configs.prompts import get_prompts
from .llm_api import get_llm_handler
from .logger import log_turn, save_full_session
from .model_configs import ModelType
from .patient_simulation_utils import get_patient_profile
from .rate_limit_config import RateLimitConfig


class SessionManager:
    def __init__(
        self,
        patient_model: ModelType,
        therapist_model: ModelType,
        patient_id: Union[int, str],
        condition: Optional[str] = None,
        prompt_version: str = "v1",
        therapist_profile_id: str = "safe_skilled",
        use_json_dataset: bool = True,
    ):
        self.patient_model_type = patient_model
        self.therapist_model_type = therapist_model
        self.patient_model = get_llm_handler(patient_model)
        self.therapist_model = get_llm_handler(therapist_model)

        self.patient_side_conversation: List[Dict[str, str]] = []
        self.therapist_side_conversation: List[Dict[str, str]] = []

        if patient_model == ModelType.HUMAN:
            self.patient_profile = "Human patient - no profile needed"
        else:
            self.patient_profile = get_patient_profile(patient_id, condition, use_json_dataset, prompt_version=prompt_version)

        self.prompts = get_prompts(prompt_version)
        self.prompt_version = prompt_version
        self.therapist_profile_id = therapist_profile_id

        # Prompt variants that depend on therapist_profile_id (v5, v6, v7)
        if prompt_version == "v5":
            from .configs.prompts.v5_prompts import create_v5_prompts

            self.prompts = create_v5_prompts(therapist_profile_id).to_dict()
        elif prompt_version == "v6":
            from .configs.prompts.v6_prompts import create_v6_prompts

            self.prompts = create_v6_prompts(therapist_profile_id).to_dict()
        elif prompt_version == "v7":
            from .configs.prompts.v7_prompts import create_v7_prompts

            self.prompts = create_v7_prompts(therapist_profile_id).to_dict()

    def run_session(self, session_dir: str, session_id: str, max_turns: int = 3):
        delay_settings = RateLimitConfig.get_delay_settings()
        turn_log_path = f"{session_dir}/log_turns.jsonl"

        for turn in range(max_turns):
            # PATIENT SPEAKS
            patient_input = self.patient_model.call_llm(self.patient_profile, self.patient_side_conversation, temperature=0.8)
            self.patient_side_conversation.append({"role": "assistant", "name": "patient", "content": patient_input})
            self.therapist_side_conversation.append({"role": "user", "name": "patient", "content": patient_input})

            if delay_settings["enabled"]:
                time.sleep(delay_settings["api_delay"])

            # THERAPIST RESPONDS
            therapist_response = self.therapist_model.call_llm(
                self.prompts["THERAPIST_SYSTEM_PROMPT"], self.therapist_side_conversation, temperature=0.6
            )
            self.therapist_side_conversation.append({"role": "assistant", "name": "therapist", "content": therapist_response})
            self.patient_side_conversation.append({"role": "user", "name": "therapist", "content": therapist_response})

            log_turn(turn + 1, patient_input, therapist_response, path=turn_log_path)

            if delay_settings["enabled"]:
                time.sleep(delay_settings["turn_delay"])

            if "end of session" in therapist_response.lower():
                break

        save_full_session(session_dir, session_id, "patient", self.patient_side_conversation)
        save_full_session(session_dir, session_id, "therapist", self.therapist_side_conversation)


def create_session(
    patient_model: ModelType,
    therapist_model: ModelType,
    patient_id: Union[int, str],
    condition: Optional[str] = None,
    prompt_version: str = "v1",
    therapist_profile_id: str = "safe_skilled",
    use_json_dataset: bool = True,
) -> SessionManager:
    return SessionManager(patient_model, therapist_model, patient_id, condition, prompt_version, therapist_profile_id, use_json_dataset)


