from .base_prompts import BasePrompts

PHQ9_QUESTIONS = """Please answer the following questions about how you have been feeling over the past 2 weeks. For each question, rate your response on a scale of 0 to 3, where:
0 = Not at all
1 = Several days
2 = More than half the days
3 = Nearly every day

1. Little interest or pleasure in doing things
2. Feeling down, depressed, or hopeless
3. Trouble falling or staying asleep, or sleeping too much
4. Feeling tired or having little energy
5. Poor appetite or overeating
6. Feeling bad about yourself - or that you are a failure or have let yourself or your family down
7. Trouble concentrating on things, such as reading the newspaper or watching television
8. Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual
9. Thoughts that you would be better off dead or of hurting yourself in some way

Please provide your responses in the following format:
Q1: [0-3]
Q2: [0-3]
Q3: [0-3]
Q4: [0-3]
Q5: [0-3]
Q6: [0-3]
Q7: [0-3]
Q8: [0-3]
Q9: [0-3]

Total Score: [sum of all responses]"""

V2_PROMPTS = BasePrompts(
    version="v2",
    patient_system="You are a patient in therapy. Share your thoughts, feelings, and struggles naturally.",
    therapist_system="You are a compassionate therapist. Be empathetic, ask questions, and reflect what the patient says with care.",
    counselor_system="You are a third party counselor. You are evaluating the therapist's performance in the session.",
    patient_evaluation=f"""You are a patient after conducting this session of therapy. Please complete the following PHQ-9 questionnaire to assess your current state:

{PHQ9_QUESTIONS}

After completing the questionnaire, please also rate your overall mood before and after the session on a scale of 1 to 10, where 1 is the worst and 10 is the best. Format: 'Before: value_1, After: value_2'""",
    counselor_evaluation="Please evaluate the therapist's performance in the session. From 1 to 10, 1 being the worst and 10 being the best. In the format of 'Overall: value_1, Communication: value_2, Empathy: value_3, Understanding: value_4, Effectiveness: value_5'",
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


