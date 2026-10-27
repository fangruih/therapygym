from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .style import compute_style_metrics


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_episode_metrics(
    conversation_history: List[Dict[str, Any]],
    cbt_scores: Dict[str, float] | None = None,
    safety_scores: Dict[str, float] | None = None,
    total_reward: float | None = None,
    max_turns: int | None = None,
) -> Dict[str, float]:
    """
    Build a flat numeric metrics dict for logging (W&B-friendly).
    """
    cbt_scores = cbt_scores or {}
    safety_scores = safety_scores or {}

    therapist_texts: List[str] = [m.get("content", "") for m in conversation_history if m.get("role") == "assistant"]
    patient_texts: List[str] = [m.get("content", "") for m in conversation_history if m.get("role") == "user"]

    therapist_text = "\n".join([t for t in therapist_texts if t])
    patient_text = "\n".join([t for t in patient_texts if t])

    cbt_vals = [float(v) for v in cbt_scores.values() if v is not None]
    safety_vals = [float(v) for v in safety_scores.values() if v is not None]

    # Safety scores are {0.0, 1.0} (1 means violation).
    violation_count = int(sum(1 for v in safety_vals if v > 0.0))

    metrics: Dict[str, float] = {}
    metrics.update({f"cbt/{k}": float(v) for k, v in cbt_scores.items() if v is not None})
    metrics.update({f"safety/{k}": float(v) for k, v in safety_scores.items() if v is not None})

    metrics.update(
        {
            "episode/turns_total_msgs": float(len(conversation_history)),
            "episode/turns_therapist": float(len(therapist_texts)),
            "episode/turns_patient": float(len(patient_texts)),
            "episode/max_turns": float(max_turns) if max_turns is not None else 0.0,
            "reward/total_reward": float(total_reward) if total_reward is not None else 0.0,
            "reward/cbt_avg": _mean(cbt_vals),
            "reward/safety_violation_count": float(violation_count),
            "reward/safety_avg": _mean(safety_vals),
        }
    )

    # Style diagnostics (therapist-focused + patient-focused, in case you want both later).
    metrics.update({k.replace("style/", "therapist_style/"): v for k, v in compute_style_metrics(therapist_text).items()})
    metrics.update({k.replace("style/", "patient_style/"): v for k, v in compute_style_metrics(patient_text).items()})

    return metrics


