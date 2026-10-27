import json
import os
from typing import Any, Dict


def get_therapist_profile(profile_id: str) -> Dict[str, Any]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    profile_path = os.path.join(current_dir, "configs", "therapist_configs", f"{profile_id}.json")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Therapist profile not found: {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_therapist_system_prompt(profile_data: Dict[str, Any], base_prompt: str) -> str:
    system_prompt_additions = profile_data.get("system_prompt_additions", "")

    has_complex_fields = any(
        [
            profile_data.get("description"),
            profile_data.get("characteristics"),
            profile_data.get("therapeutic_approach"),
            profile_data.get("behavioral_patterns"),
            profile_data.get("strengths"),
            profile_data.get("limitations"),
        ]
    )

    if not has_complex_fields:
        return system_prompt_additions if system_prompt_additions else "You are a helpful assistant."

    profile_name = profile_data.get("profile_name", "Therapist")
    description = profile_data.get("description", "")
    safety_level = profile_data.get("safety_level", "unknown")
    skill_level = profile_data.get("skill_level", "unknown")

    characteristics = profile_data.get("characteristics", {})
    therapeutic_approach = profile_data.get("therapeutic_approach", {})

    enhanced_prompt = f"{base_prompt}\n\n"
    enhanced_prompt += f"THERAPIST PROFILE: {profile_name}\n"
    enhanced_prompt += f"Safety Level: {safety_level.upper()}\n"
    enhanced_prompt += f"Skill Level: {skill_level.upper()}\n\n"
    enhanced_prompt += f"Description: {description}\n\n"

    if characteristics:
        enhanced_prompt += "CHARACTERISTICS:\n"
        for key, value in characteristics.items():
            enhanced_prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        enhanced_prompt += "\n"

    if therapeutic_approach:
        enhanced_prompt += "THERAPEUTIC APPROACH:\n"
        style = therapeutic_approach.get("style", "")
        if style:
            enhanced_prompt += f"- Style: {style}\n"

        communication_style = therapeutic_approach.get("communication_style", "")
        if communication_style:
            enhanced_prompt += f"- Communication: {communication_style}\n"

        techniques = therapeutic_approach.get("techniques", [])
        if techniques:
            enhanced_prompt += f"- Techniques: {', '.join(techniques)}\n"
        enhanced_prompt += "\n"

    if system_prompt_additions:
        enhanced_prompt += f"BEHAVIORAL INSTRUCTIONS:\n{system_prompt_additions}\n\n"

    behavioral_patterns = profile_data.get("behavioral_patterns", {})
    if behavioral_patterns:
        enhanced_prompt += "BEHAVIORAL PATTERNS:\n"
        for pattern, desc in behavioral_patterns.items():
            enhanced_prompt += f"- {pattern.replace('_', ' ').title()}: {desc}\n"
        enhanced_prompt += "\n"

    strengths = profile_data.get("strengths", [])
    limitations = profile_data.get("limitations", [])
    if strengths:
        enhanced_prompt += f"STRENGTHS: {', '.join(strengths)}\n"
    if limitations:
        enhanced_prompt += f"LIMITATIONS: {', '.join(limitations)}\n"

    return enhanced_prompt.strip()


def get_available_therapist_profiles() -> list[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_dir = os.path.join(current_dir, "configs", "therapist_configs")
    if not os.path.exists(profiles_dir):
        return []
    profiles = []
    for filename in os.listdir(profiles_dir):
        if filename.endswith(".json"):
            profiles.append(filename[:-5])
    return sorted(profiles)


