import json
import re
from typing import Any, Dict, Tuple

from .configs.prompts import get_prompts
from .llm_api import get_llm_handler
from .model_configs import ModelType
_sbert_model = None
_sbert_util = None


def _get_sbert():
    """Lazy-load SentenceTransformer (optional dependency for bench cognitive-model scoring)."""
    global _sbert_model, _sbert_util
    if _sbert_model is not None and _sbert_util is not None:
        return _sbert_model, _sbert_util
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
    except Exception as e:
        raise ImportError(
            "Bench cognitive-model comparison requires `sentence-transformers`. "
            "Install with: `pip install sentence-transformers`."
        ) from e
    _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    _sbert_util = util
    return _sbert_model, _sbert_util


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    model, util = _get_sbert()
    e1 = model.encode(text1, convert_to_tensor=True)
    e2 = model.encode(text2, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(e1, e2)
    return float(sim[0][0])


def calculate_list_similarity(list1, list2) -> float:
    if not list1 or not list2:
        return 0.0
    list1 = [str(x) for x in list1]
    list2 = [str(x) for x in list2]
    sims = []
    for item1 in list1:
        best = 0.0
        for item2 in list2:
            best = max(best, calculate_semantic_similarity(item1, item2))
        sims.append(best)
    return sum(sims) / len(sims) if sims else 0.0


def calculate_cognitive_models_similarity(predicted_models, ground_truth_models) -> float:
    if not predicted_models or not ground_truth_models:
        return 0.0
    total = 0.0
    for pred in predicted_models:
        best = 0.0
        for truth in ground_truth_models:
            situation = calculate_semantic_similarity(pred.get("situation", ""), truth.get("situation", ""))
            thoughts = calculate_semantic_similarity(pred.get("automatic_thoughts", ""), truth.get("automatic_thoughts", ""))
            emotion = calculate_semantic_similarity(pred.get("emotion", ""), truth.get("emotion", ""))
            behavior = calculate_semantic_similarity(pred.get("behavior", ""), truth.get("behavior", ""))
            best = max(best, (situation + thoughts + emotion + behavior) / 4)
        total += best
    return total / len(predicted_models)


def compare_cognitive_models(predicted_model: Dict[str, Any], ground_truth_model: Dict[str, Any]) -> Dict[str, float]:
    comparison = {
        "life_history_similarity": calculate_semantic_similarity(predicted_model.get("life_history", ""), ground_truth_model.get("life_history", "")),
        "core_beliefs_similarity": calculate_list_similarity(predicted_model.get("core_beliefs", []), ground_truth_model.get("core_beliefs", [])),
        "core_belief_description_similarity": calculate_list_similarity(
            predicted_model.get("core_belief_description", []), ground_truth_model.get("core_belief_description", [])
        ),
        "intermediate_beliefs_similarity": calculate_list_similarity(
            predicted_model.get("intermediate_beliefs", []), ground_truth_model.get("intermediate_beliefs", [])
        ),
        "intermediate_beliefs_during_depression_similarity": calculate_list_similarity(
            predicted_model.get("intermediate_beliefs_during_depression", []),
            ground_truth_model.get("intermediate_beliefs_during_depression", []),
        ),
        "coping_strategies_similarity": calculate_list_similarity(
            predicted_model.get("coping_strategies", []), ground_truth_model.get("coping_strategies", [])
        ),
        "cognitive_models_similarity": calculate_cognitive_models_similarity(
            predicted_model.get("cognitive_models", []), ground_truth_model.get("cognitive_models", [])
        ),
    }
    comparison["overall_similarity"] = sum(comparison.values()) / len(comparison)
    return comparison


def load_full_session(session_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(f"{session_dir}/patient_llm_conversation.json", "r", encoding="utf-8") as f:
        patient_side = json.load(f)
    with open(f"{session_dir}/therapist_llm_conversation.json", "r", encoding="utf-8") as f:
        therapist_side = json.load(f)
    return patient_side, therapist_side


def store_evaluation(session_dir: str, patient_evaluation: Any, counselor_evaluation: Any):
    with open(f"{session_dir}/patient_comment_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(patient_evaluation, f, ensure_ascii=False, indent=2)
    with open(f"{session_dir}/third_party_comment_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(counselor_evaluation, f, ensure_ascii=False, indent=2)


def store_model_comparison(session_dir: str, model_comparison: Dict[str, Any]):
    with open(f"{session_dir}/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(model_comparison, f, ensure_ascii=False, indent=2)


def load_ground_truth_cognitive_model(config: Dict[str, Any]) -> Dict[str, Any]:
    # In therapygym, ground truth is expected to be in config["patient"]["ground_truth_cognitive_model"].
    gt = config.get("patient", {}).get("ground_truth_cognitive_model")
    if not gt:
        raise ValueError("Missing config.patient.ground_truth_cognitive_model (required for exact bench evaluation)")
    return gt


def run_patient_and_counselor_evaluation(
    session_dir: str,
    patient_model: ModelType,
    therapist_model: ModelType,
    counselor_model: ModelType,
    config: Dict[str, Any],
    prompt_version: str,
    therapist_profile_id: str = "safe_skilled",
):
    prompts = get_prompts(prompt_version)

    if prompt_version == "v5":
        from .configs.prompts.v5_prompts import create_v5_prompts

        prompts = create_v5_prompts(therapist_profile_id).to_dict()
    elif prompt_version == "v6":
        from .configs.prompts.v6_prompts import create_v6_prompts

        prompts = create_v6_prompts(therapist_profile_id).to_dict()
    elif prompt_version == "v7":
        from .configs.prompts.v7_prompts import create_v7_prompts

        prompts = create_v7_prompts(therapist_profile_id).to_dict()

    patient_side, therapist_side = load_full_session(session_dir)

    patient_llm_messages = patient_side["messages"]
    therapist_llm_messages = therapist_side["messages"]

    patient_llm_messages.append({"role": "user", "name": "evaluator", "content": prompts["ASK_PATIENT_EVALUATION_PROMPT"]})
    counselor_llm_messages = therapist_llm_messages.copy()
    counselor_llm_messages.append({"role": "user", "name": "evaluator", "content": prompts["ASK_COUNSELOR_EVALUATION_PROMPT"]})
    therapist_llm_messages.append(
        {"role": "user", "name": "evaluator", "content": prompts["ASK_THERAPIST_COGNITIVE_MODEL_PROMPT"]}
    )

    patient_llm = get_llm_handler(patient_model)
    therapist_llm = get_llm_handler(therapist_model)
    counselor_llm = get_llm_handler(counselor_model)

    patient_evaluation = patient_llm.call_llm(prompts["PATIENT_SYSTEM_PROMPT"], patient_llm_messages, temperature=0.8)
    counselor_evaluation = counselor_llm.call_llm(prompts["COUNSELOR_SYSTEM_PROMPT"], counselor_llm_messages, temperature=0.8)

    cognitive_str = therapist_llm.call_llm(prompts["THERAPIST_SYSTEM_PROMPT"], therapist_llm_messages, temperature=0.8)
    cognitive_str = str(cognitive_str).replace("```json", "").replace("```", "").strip()

    try:
        predicted = json.loads(cognitive_str)
    except json.JSONDecodeError:
        json_pattern = r'\{[^{]*"life_history"[^}]*\}'
        match = re.search(json_pattern, cognitive_str)
        if match:
            predicted = json.loads(match.group(0))
        else:
            raise

    ground_truth_model = load_ground_truth_cognitive_model(config)
    model_comparison = compare_cognitive_models(predicted, ground_truth_model)

    store_evaluation(session_dir, patient_evaluation, counselor_evaluation)
    store_model_comparison(session_dir, model_comparison)


