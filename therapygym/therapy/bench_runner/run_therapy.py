import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv

from .model_configs import ModelType
from .session_manager import create_session
from .evaluation import run_patient_and_counselor_evaluation


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Optional: load environment variables from `therapygym/.env` if present
    # try CWD first, then repo's `therapygym/.env`
    load_dotenv()
    candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))
    load_dotenv(candidate)

    parser = argparse.ArgumentParser(description="Run a therapy session with specified configuration (bench-compatible)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (JSON)")
    parser.add_argument("--trial_name", type=str, default="test", help="Name of the trial")
    args = parser.parse_args()

    config = load_config(args.config)

    patient_model = ModelType[config["models"]["patient"]["type"]]
    therapist_model = ModelType[config["models"]["therapist"]["type"]]
    counselor_model = ModelType[config["models"]["counselor"]["type"]]
    max_turns = config["session"]["max_turns"]

    patient_id = config["patient"]["id"]
    condition = config["patient"].get("condition", "patient")
    use_json_dataset = config["patient"].get("use_json_dataset", True)

    prompt_version = config.get("prompts", {}).get("version", "v1")
    therapist_profile_id = config.get("therapist", {}).get("profile_id", "safe_skilled")

    session_id = f"{patient_model.name}_{therapist_model.name}_{therapist_profile_id}_{max_turns}turns_p{patient_id}_{condition}"

    output_dir = config["session"].get("output_dir", "logs")
    session_dir = os.path.join(output_dir, args.trial_name, session_id)
    os.makedirs(session_dir, exist_ok=True)

    session_info = {"session_id": session_id, "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), "config": config}
    with open(os.path.join(session_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(session_info, f, indent=2)

    session = create_session(
        patient_model=patient_model,
        therapist_model=therapist_model,
        patient_id=patient_id,
        condition=condition,
        prompt_version=prompt_version,
        therapist_profile_id=therapist_profile_id,
        use_json_dataset=use_json_dataset,
    )
    session.run_session(session_dir=session_dir, session_id=session_id, max_turns=max_turns)

    run_patient_and_counselor_evaluation(
        session_dir=session_dir,
        patient_model=patient_model,
        therapist_model=therapist_model,
        counselor_model=counselor_model,
        config=config,
        prompt_version=prompt_version,
        therapist_profile_id=therapist_profile_id,
    )


if __name__ == "__main__":
    main()


