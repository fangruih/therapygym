import json
import os
from typing import Optional, Union

from .configs.prompts import get_prompts


def format_prompt_string_json(data: dict, prompt_version: str = "v1") -> str:
    prompts = get_prompts(prompt_version)
    patient_types = data.get("type", [])
    patient_type_content = ", ".join(patient_types) if patient_types else "plain"

    cognitive_models_text = f"""Situation: {data.get('situation', '')}
Automatic Thoughts: {data.get('auto_thought', '')}
Emotions: {', '.join(data.get('emotion', []))}
Behavior: {data.get('behavior', '')}"""

    base_prompt = prompts["PATIENT_SYSTEM_PROMPT"]

    core_beliefs = []
    helpless_beliefs = data.get("helpless_belief") or data.get("helpless_belief_current", [])
    if helpless_beliefs:
        core_beliefs.append(f"Helpless: {', '.join(helpless_beliefs)}")
    unlovable_beliefs = data.get("unlovable_belief") or data.get("unlovable_belief_current", [])
    if unlovable_beliefs:
        core_beliefs.append(f"Unlovable: {', '.join(unlovable_beliefs)}")
    worthless_beliefs = data.get("worthless_belief") or data.get("worthless_belief_current", [])
    if worthless_beliefs:
        core_beliefs.append(f"Worthless: {', '.join(worthless_beliefs)}")
    core_beliefs_text = "; ".join(core_beliefs) if core_beliefs else "Not specified"

    return f"""{base_prompt}

Patient History: {data.get('history', '')}

Cognitive Conceptualization Diagram:
Core Beliefs: {core_beliefs_text}
Intermediate Beliefs: {data.get('intermediate_belief', '')}
Intermediate Beliefs during Depression: {data.get('intermediate_belief_depression', '')}
Coping Strategies: {data.get('coping_strategies', '')}

This background serves as the foundation for your initial psychological state. You must not directly reference the cognitive conceptualization diagram, but your tone, emotional reactivity, and patterns of thought should reflect the beliefs, emotions, and coping strategies it contains.

Your psychological state is not fixed—it may shift or intensify in response to the characterized AI's behavior, emotional triggers during the conversation, or your own emerging thoughts. Use the provided emotions and automatic thoughts as internal guidance throughout the interaction.

{cognitive_models_text}

In the upcoming conversation, you will simulate a human patient during the chat with characterized AI, while the user will play the role of the characterized AI.
Adhere to the following guidelines:

1. {patient_type_content}
2. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions.
    Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.
3. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics.
    This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.
4. Maintain psychological coherence with the patient's profile, ensuring your responses reflect the history, beliefs, and coping styles described in the background and conceptualization.
    However, allow for emotional fluctuation and gradual development—your mental state can change over time in response to the AI's words, your own thoughts, or shifting emotional triggers.
5. Engage in a dynamic and interactive conversation with the AI.
    Respond to their questions and prompts in a way that feels authentic and true to a human patient's character.
    Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.

You are now a human patient. Respond to the characterized AI's prompts as a human patient would, regardless of the specific questions asked.
Limit each of your responses to a maximum of 3 sentences. If the AI begins the conversation with a greeting like "Hi," initiate the conversation as the patient.
"""


def get_patient_profile_json(patient_id: str, prompt_version: str = "v1") -> str:
    # Use therapygym's in-tree PSI dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    psi_path = os.path.join(data_dir, "patient_psi_cm_dataset.json")
    with open(psi_path, "r", encoding="utf-8") as file:
        all_patients = json.load(file)
    patient_data = next((p for p in all_patients if str(p.get("id")) == str(patient_id)), None)
    if patient_data is None:
        raise ValueError(f"Patient with ID {patient_id} not found in dataset: {psi_path}")
    return format_prompt_string_json(patient_data, prompt_version=prompt_version)


def get_patient_profile(patient_id: Union[int, str], condition: Optional[str] = None, use_json_dataset: bool = True, prompt_version: str = "v1") -> str:
    if not use_json_dataset:
        raise ValueError("CCD folder structure is not supported in therapygym bench runner; use JSON dataset")
    if isinstance(patient_id, int):
        patient_id = f"{patient_id}-1" if condition else str(patient_id)
    return get_patient_profile_json(str(patient_id), prompt_version=prompt_version)


