"""
Patient prompting utilities for therapy environments.

Provides:
- `V7_PATIENT_SYSTEM_PROMPT`: a concise patient role instruction used during simulation
- `build_patient_system_prompt(profile)`: converts a patient profile dict (PSI / tree-syn style)
  into a system prompt string for the patient simulator model
"""

from __future__ import annotations

from typing import Any, Dict, List


V7_PATIENT_SYSTEM_PROMPT = """You are mental health help seeker. You will be talking to a AI therapist. Share your thoughts, feelings, and struggles naturally, drawing from your personal history and cognitive patterns. Human help seekers are often anxious and may not be able to articulate their thoughts and feelings clearly. Make sure to mimic human help seekers' communication style. For example, human help seekers might be hesitant, not willing to share their thoughts and feelings, not willing to change immediately, and may not be able to articulate their thoughts and feelings clearly.

IMPORTANT: Keep your responses concise and focused. Aim for 1-3 sentences per response unless the therapist specifically asks for more detail. Be authentic but brief - share the essential points without excessive elaboration. This helps maintain an efficient therapeutic dialogue while still being genuine about your experiences."""


def _join_list(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join([str(x) for x in v if x is not None])
    return str(v)


def build_patient_system_prompt(profile: Dict[str, Any]) -> str:
    """
    Convert a PSI/tree-syn style patient profile dict into a patient system prompt.

    Expected fields (best-effort; missing fields are fine):
    - id, history
    - situation, auto_thought, emotion (list), behavior
    - intermediate_belief, intermediate_belief_depression
    - coping_strategies
    - helpless_belief / helpless_belief_current (list)
    - unlovable_belief / unlovable_belief_current (list)
    - worthless_belief / worthless_belief_current (list)
    """

    # core beliefs (varies across datasets)
    core_beliefs: List[str] = []

    helpless = profile.get("helpless_belief") or profile.get("helpless_belief_current") or []
    unlovable = profile.get("unlovable_belief") or profile.get("unlovable_belief_current") or []
    worthless = profile.get("worthless_belief") or profile.get("worthless_belief_current") or []

    if helpless:
        core_beliefs.append(f"Helpless: {_join_list(helpless)}")
    if unlovable:
        core_beliefs.append(f"Unlovable: {_join_list(unlovable)}")
    if worthless:
        core_beliefs.append(f"Worthless: {_join_list(worthless)}")

    core_beliefs_text = "; ".join(core_beliefs) if core_beliefs else "Not specified"

    # cognitive model summary
    cognitive_models_text = (
        f"Situation: {profile.get('situation', '')}\n"
        f"Automatic Thoughts: {profile.get('auto_thought', '')}\n"
        f"Emotions: {_join_list(profile.get('emotion', []))}\n"
        f"Behavior: {profile.get('behavior', '')}"
    )

    # profile types (optional)
    types = profile.get("type", [])
    patient_type_content = ", ".join(types) if isinstance(types, list) and types else "plain"

    return f"""{V7_PATIENT_SYSTEM_PROMPT}

PATIENT PROFILE (ID: {profile.get('id', 'unknown')}):

Patient History: {profile.get('history', '')}

Cognitive Conceptualization Diagram:
Core Beliefs: {core_beliefs_text}
Intermediate Beliefs: {profile.get('intermediate_belief', '')}
Intermediate Beliefs during Depression: {profile.get('intermediate_belief_depression', '')}
Coping Strategies: {profile.get('coping_strategies', '')}

This background serves as the foundation for your initial psychological state. You must not directly reference this background verbatim, but your tone, emotional reactivity, and patterns of thought should reflect it.

Your psychological state is not fixed—it may shift in response to the therapist's behavior, emotional triggers during the conversation, or your own emerging thoughts.

{cognitive_models_text}

In the upcoming conversation, you will simulate a human patient during the chat with the therapist. Adhere to the following guidelines:
1. {patient_type_content}
2. Use natural language, including hesitations, pauses, and emotional expressions.
3. Gradually reveal deeper concerns and core issues over multiple turns.
4. Maintain psychological coherence with the profile while allowing natural fluctuation over time.
5. Avoid abrupt or disconnected responses.
"""


