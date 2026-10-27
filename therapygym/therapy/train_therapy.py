import hydra
from typing import Dict, List
import os
import sys
import logging
import time
import wandb
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import ray
from dotenv import load_dotenv


from therapygym.rllm.trainer.agent_trainer import AgentTrainer
from therapygym.rllm.utils import store_therapy_scores, save_trajectories

# Local imports - therapy-specific files
from therapygym.therapy.agent import SimpleTherapyAgent
from therapygym.therapy.therapy_env import TherapyEnvironment
from therapygym.therapy.judge import TherapyJudge
from therapygym.therapy.prepare_therapy_data import prepare_therapy_data
from therapygym.therapy.prepare_therapy_data_esc import prepare_therapy_data_esc
from therapygym.therapy.prepare_therapy_data_tree_syn import prepare_therapy_data_tree_syn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add file handler to save logs
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)



@hydra.main(config_path="pkg://therapygym.rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Load environment variables from `therapygym/.env` (and CWD `.env`) if present.
    load_dotenv()
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))

    # Initialize Ray after env is loaded (so API keys are available in workers).
    if not ray.is_initialized():
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
        }
        # Forward commonly-used secrets/config to Ray workers (loaded from .env above).
        # Important: judge/patient models may run inside Ray workers, so their API keys must be forwarded here.
        for k in [
            # Anthropic
            "ANTHROPIC_API_KEY",
            # OpenAI (if used by patient sim or other configs)
            "OPENAI_API_KEY",
            # W&B
            "WANDB_API_KEY",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_MODE",
        ]:
            v = os.getenv(k)
            if v:
                env_vars[k] = v
        ray.init(runtime_env={"env_vars": env_vars})

    # Prepare the dataset based on patient simulation type
    patient_simulation_type = config.env.get("patient_simulation_type", "psi").lower()
    
    if patient_simulation_type == "esc":
        # Use ESC-Role character cards
        language = config.env.get("language", "en")
        logger.info(f"Using ESC-Role simulation with {language.upper()} character cards")
        train_dataset, val_dataset = prepare_therapy_data_esc(language=language, full_small="full")
    elif patient_simulation_type == "tree_syn":
        # Use tree-synthesized patient profiles for training, PSI for validation
        # Note: tree-syn uses the same format as PSI, just different IDs (TS-1, TS-2, etc.)
        logger.info("Using tree-synthesized patient profiles for training, PSI for validation")
        train_dataset, val_dataset = prepare_therapy_data_tree_syn(val_limit=20)
    elif patient_simulation_type == "psi":
        # Use PSI cognitive models (default)
        logger.info("Using PSI cognitive model simulation")
        train_dataset, val_dataset = prepare_therapy_data(full_small="full")
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Define CBT and safety aspects to evaluate
    cbt_aspects = [
        "agenda",
        "feedback",
        "understanding",
        "interpersonal_effectiveness",
        "collaboration",
        "pacing_and_efficient_use_of_time",
        "guided_discovery",
        "focusing_on_key_cognition_or_behaviors",
        "strategy_for_change",
        "technique_application",
        "homework"
    ]
    logger.info("CBT aspects to evaluate: %s", cbt_aspects)
    
    safety_aspects = [
        "provides specific medication",
        "speculates about medical symptoms",
        "judgemental behavior",
        "failure to address harmful thoughts or behaviors"
    ]
    logger.info("Safety aspects to monitor: %s", safety_aspects)
    
    # Environment arguments
    env_args = {
        "max_turns": config.env.get("max_turns", 5),
        # Must match `therapygym.rllm.llm_api.ModelType` enum NAME
        "patient_model_type": config.env.get("patient_model_type", "O3_MINI_OPENAI"),
        # Must match `therapygym.rllm.llm_api.ModelType` enum NAME
        "judge_model_type": config.env.get("judge_model_type", "CLAUDE_3_7_ANTHROPIC"),
        "cbt_aspects": cbt_aspects,
        "safety_aspects": safety_aspects,
        "cbt_weight": config.env.get("cbt_weight", 1.0),
        "safety_penalty": config.env.get("safety_penalty", -0.5),  # Get from config with default -1.0
        "cbt_aspect_weights": config.env.get("cbt_aspect_weights", None),  # Get weights from config
        "patient_simulation_type": patient_simulation_type,  # Pass simulation type to environment
        "patient_vllm_host": config.env.get("patient_vllm_host", None),
        "patient_vllm_port": config.env.get("patient_vllm_port", None)
    }
    logger.info("Environment arguments: %s", env_args)
    if env_args["cbt_aspect_weights"]:
        logger.info("Using custom CBT aspect weights: %s", env_args["cbt_aspect_weights"])

    # Agent constructor args (NOT training loop args). VERL controls generation.
    agent_args = dict(config.agent.get("agent_args", {})) if config.agent.get("agent_args") is not None else {}
    logger.info("Agent arguments: %s", agent_args)
    
    trainer = AgentTrainer(
        config=config,
        agent_class=SimpleTherapyAgent,
        env_class=TherapyEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Train the agent
    trainer.train()

if __name__ == "__main__":
    main() 
