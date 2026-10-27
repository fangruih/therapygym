import json
from datetime import datetime
import os


def log_turn(turn_num, patient_msg, therapist_msg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        entry = {"turn": turn_num, "timestamp": datetime.now().isoformat(), "patient": patient_msg, "therapist": therapist_msg}
        f.write(json.dumps(entry) + "\n")


def save_full_session(session_dir: str, session_id: str, patient_or_therapist: str, conversation):
    session_data = {"session_id": session_id, "timestamp": datetime.now().isoformat(), "messages": conversation}
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, f"{patient_or_therapist}_llm_conversation.json"), "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)


