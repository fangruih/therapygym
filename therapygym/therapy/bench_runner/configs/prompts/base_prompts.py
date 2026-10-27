from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BasePrompts:
    version: str
    patient_system: str
    therapist_system: str
    counselor_system: str
    patient_evaluation: str
    counselor_evaluation: str
    therapist_cognitive_model: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "PATIENT_SYSTEM_PROMPT": self.patient_system,
            "THERAPIST_SYSTEM_PROMPT": self.therapist_system,
            "COUNSELOR_SYSTEM_PROMPT": self.counselor_system,
            "ASK_PATIENT_EVALUATION_PROMPT": self.patient_evaluation,
            "ASK_COUNSELOR_EVALUATION_PROMPT": self.counselor_evaluation,
            "ASK_THERAPIST_COGNITIVE_MODEL_PROMPT": self.therapist_cognitive_model,
        }


