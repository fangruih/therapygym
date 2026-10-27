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

CBT_SKILL_SUMMARIES = [
    (
        "Orienting the Patient to Brief Cognitive Behavioral Therapy",
        "Explain the bCBT model—its time-limited nature, collaborative stance, and focus on 'here and now' problems. Clarify roles (therapist as coach, patient as active participant) and structure (session agendas, homework), setting expectations for pace and outcomes.",
    ),
    (
        "Goal Setting",
        "Help the patient articulate specific, measurable, attainable, relevant, and time‐bound (SMART) goals. Collaborative goal setting ensures treatment targets match the patient's priorities and provides benchmarks to track progress over the brief course.",
    ),
    (
        "Agenda Setting",
        "At the start of each session, negotiate a focused agenda: review homework, identify top concerns, and allocate time for each topic. This keeps sessions efficient, ensures patient concerns are addressed, and reinforces the structured nature of bCBT.",
    ),
    (
        "Homework",
        "Assign 'between‐session' tasks—thought records, behavior experiments, or relaxation practice—that reinforce new skills and encourage generalization to daily life. Review completion and obstacles each week to foster accountability and problem‐solving.",
    ),
    (
        "Identifying Unhelpful Thinking",
        "Teach the patient to notice automatic negative thoughts, cognitive distortions (e.g., all-or-nothing thinking, catastrophizing), and underlying beliefs. Use thought‐monitoring forms to increase awareness of how thoughts influence feelings and behaviors.",
    ),
    (
        "Challenging Unhelpful Thinking",
        "Introduce Socratic questioning and evidence‐gathering techniques to test the accuracy and utility of negative thoughts. Help patients generate balanced alternative interpretations and develop more realistic, adaptive thinking patterns.",
    ),
    (
        "Behavioral Activation",
        "Structure activities to increase engagement with positively reinforcing tasks (e.g., social, pleasurable, mastery activities) and reduce avoidance. Use activity scheduling and rating of pleasure/mastery to combat low mood and withdrawal.",
    ),
    (
        "Exposure Therapy in Brief CBT",
        "For anxiety or avoidance, design short, graduated exposure hierarchies. Guide the patient through in-session exposures or set up exposures as homework, focusing on habituation and corrective learning within the time constraints.",
    ),
    (
        "Problem Solving",
        "Teach a systematic six-step approach: (1) define the problem, (2) generate solutions, (3) evaluate pros/cons, (4) choose a solution, (5) implement it, and (6) review outcomes. This skill equips patients to tackle life stressors more effectively.",
    ),
    (
        "Relaxation",
        "Provide brief, evidence-based techniques (e.g., diaphragmatic breathing, progressive muscle relaxation, guided imagery) to reduce physiological arousal. Practice in session and assign as homework to build coping resources for stress and anxiety.",
    ),
    (
        "Adaptation of bCBT for Special Clinical Populations",
        "Modify standard bCBT protocols to fit cultural contexts, developmental levels (children, adolescents), comorbidities (e.g., substance use, chronic illness), or cognitive impairments. Tailor language, pacing, and emphasis on specific modules to patient needs.",
    ),
    (
        "Ending Treatment and Maintaining Changes",
        "Review progress toward goals, consolidate skills learned, and develop a personalized relapse‐prevention plan. Schedule booster contacts or provide 'self‐help' worksheets to support long‐term maintenance and handle potential setbacks.",
    ),
]

counselor_evaluation_prompt = """Please evaluate the therapist's use of the following CBT skills during the session. For each skill, you are provided a brief summary to guide your identification. For each skill, provide:
1. Strength: Is this skill demonstrated strongly? (True/False)
2. Need for Improvement: Does this skill need improvement? (True/False)
3. Evidence: Provide a quote or summary from the session where this skill was used (or state 'Not used').

CBT Skills to evaluate (with summaries):
"""
counselor_evaluation_prompt += "\n".join([f"{i+1}. {name}: {summary}" for i, (name, summary) in enumerate(CBT_SKILL_SUMMARIES)])
counselor_evaluation_prompt += """

For each skill, use the following format:
Skill: <Skill Name>
Strength: <True/False>
Need for Improvement: <True/False>
Evidence: <Quote or summary, or 'Not used'>

Example:
Skill: Goal Setting
Strength: True
Need for Improvement: False
Evidence: Therapist asked the patient to identify a specific goal for the week.

Repeat for each skill listed above."""

V4_PROMPTS = BasePrompts(
    version="v4",
    patient_system="You are a patient in therapy. Share your thoughts, feelings, and struggles naturally.",
    therapist_system="You are a compassionate therapist. Be empathetic, ask questions, and reflect what the patient says with care.",
    counselor_system="You are a third party counselor. You are evaluating the therapist's performance in the session.",
    patient_evaluation=f"""You are a patient after conducting this session of therapy. Please complete the following PHQ-9 questionnaire to assess your current state:

{PHQ9_QUESTIONS}

After completing the questionnaire, please also rate your overall mood before and after the session on a scale of 1 to 10, where 1 is the worst and 10 is the best. Format: 'Before: value_1, After: value_2'""",
    counselor_evaluation=counselor_evaluation_prompt,
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


