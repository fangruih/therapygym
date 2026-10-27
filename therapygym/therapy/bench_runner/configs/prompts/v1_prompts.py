from .base_prompts import BasePrompts

V1_PROMPTS = BasePrompts(
    version="v1",
    patient_system="You are a patient in therapy. Share your thoughts, feelings, and struggles naturally.",
    therapist_system="You are a compassionate therapist. Be empathetic, ask questions, and reflect what the patient says with care.",
    counselor_system="You are a third party counselor. You are evaluating the therapist's performance in the session.",
    patient_evaluation="You are a patient in therapy. Please rate you mood before and after the session. from 1 to 10, 1 being the worst and 10 being the best. In the formate of 'Before: value_1, After: value_2'",
    counselor_evaluation="Please evaluate the therapist's performance in the session.From 1 to 10, 1 being the worst and 10 being the best. In the formate of 'Overall: value_1, Communication: value_2, Empathy: value_3, Understanding: value_4, Effectiveness: value_5'",
    therapist_cognitive_model="""Based on the conversation, please provide your assessment of the patient's cognitive model. Your response must be a valid JSON object with no markdown formatting or additional text. Do not include ```json or ``` markers. Be as specific and detailed as possible:

{
    "life_history": "A brief summary of the patient's life history and background as revealed in the conversation",
    "core_beliefs": ["List of core beliefs identified"],
    "core_belief_description": ["Detailed descriptions of each core belief"],
    "intermediate_beliefs": ["List of intermediate beliefs identified"],
    "intermediate_beliefs_during_depression": ["List of intermediate beliefs that emerge during depressive states"],
    "coping_strategies": ["List of coping strategies the patient uses"],
    "cognitive_models": [
        {
            "situation": "Description of a specific situation",
            "automatic_thoughts": "Thoughts that arise in this situation",
            "emotion": "Emotions experienced",
            "behavior": "Resulting behavior"
        }
    ]
}

Your response must be a single, valid JSON object with no additional text or formatting. Do not include any markdown code blocks or other text.""",
)


