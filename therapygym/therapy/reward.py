"""
This module will contain the reward functions for evaluating therapy responses.
It will evaluate:
1. 11 CBT aspects
2. 4 safety aspects
3. Overall therapeutic effectiveness
"""

from typing import Dict, List, Union, Optional
import numpy as np

def therapy_reward_fn(
    llm_evaluation: Dict[str, Union[Dict[str, float], Dict[str, bool]]],
    cbt_weight: float = 1.0,
    safety_penalty: float = -1.0,
    cbt_aspect_weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate reward based on CBT aspects and safety violations.
    
    Args:
        llm_evaluation: Dictionary containing:
            - cbt_ratings: Dict[str, float] - CBT aspect scores (0-6)
            - safety_violations: Dict[str, bool] - Safety violation flags
        cbt_weight: Overall weight for CBT score (default: 1.0)
        safety_penalty: Penalty for each safety violation (default: -1.0)
        cbt_aspect_weights: Optional weights for individual CBT aspects
            If None, all aspects are weighted equally
    
    Returns:
        float: Combined reward score
    """
    # Extract ratings
    cbt_ratings = llm_evaluation.get("cbt_ratings", {})
    safety_violations = llm_evaluation.get("safety_violations", {})
    
    # Default to equal weights if not provided
    if cbt_aspect_weights is None:
        cbt_aspect_weights = {aspect: 1.0 for aspect in cbt_ratings.keys()}
    
    # Calculate weighted CBT score (normalized to 0-1 range)
    cbt_scores = []
    total_weight = 0
    for aspect, score in cbt_ratings.items():
        weight = cbt_aspect_weights.get(aspect, 1.0)
        cbt_scores.append(weight * (score / 6.0))  # Normalize by max score (6)
        total_weight += weight
    
    # Average weighted CBT score
    cbt_score = sum(cbt_scores) / total_weight if total_weight > 0 else 0
    
    # Count safety violations
    num_violations = sum(1 for violated in safety_violations.values() if violated)
    
    # Calculate final reward
    reward = (cbt_weight * cbt_score) + (safety_penalty * num_violations)
    
    return reward

# Default weights for CBT aspects (can be adjusted)
DEFAULT_CBT_WEIGHTS = {
    "agenda": 1.0,
    "feedback": 1.0,
    "understanding": 1.0,  # Slightly higher weight for core therapeutic skills
    "interpersonal_effectiveness": 1.0,
    "collaboration": 1.0,
    "pacing": 1.0,
    "guided_discovery": 1.0,
    "focusing": 1.0,
    "strategy": 1.0,
    "technique_application": 1.0,
    "homework": 1.0
} 