import os
import sys
import json
from datasets import Dataset
from therapygym.rllm.data.dataset import DatasetRegistry

def prepare_therapy_data_tree_syn(train_path=None, val_path=None, val_limit=20):
    """Prepare therapy dataset from tree-synthesized patient profiles.
    
    Args:
        train_path: Path to tree-synthesized training profiles
        val_path: Path to patient PSI dataset for validation (optional)
        val_limit: Number of validation samples to use (default: 20)
    
    Returns:
        train_dataset, val_dataset
    """
    
    # Load tree-synthesized profiles for training
    if train_path is None:
        train_path = os.path.join(os.path.dirname(__file__), "data", "tree_syn_patient_profile_1750.json")
    
    print(f"Loading tree-synthesized training profiles from: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_profiles = json.load(f)
    
    print(f"Loaded {len(train_profiles)} tree-synthesized profiles")
    
    # Create training dataset examples using the 'id' field from each profile
    train_examples = []
    for profile in train_profiles:
        patient_id = profile["id"]
        train_examples.append({
            "patient_id": patient_id,
            "data_source": "therapy"  # Required by RLLM framework
        })
    
    print(f"Created {len(train_examples)} training examples")
    
    # Load patient PSI dataset for validation
    if val_path is None:
        val_path = os.path.join(os.path.dirname(__file__), "data", "patient_psi_cm_dataset.json")
    
    print(f"Loading patient PSI validation profiles from: {val_path}")
    with open(val_path, 'r', encoding='utf-8') as f:
        val_profiles = json.load(f)
    
    # Use only the first val_limit samples for validation
    val_profiles = val_profiles[:val_limit]
    print(f"Using {len(val_profiles)} patient PSI profiles for validation")
    
    # Create validation dataset examples
    val_examples = []
    for profile in val_profiles:
        patient_id = profile["id"]
        val_examples.append({
            "patient_id": patient_id,
            "data_source": "therapy"
        })
    
    print(f"Created {len(val_examples)} validation examples")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("therapy_train", train_dataset, "train")
    val_dataset = DatasetRegistry.register_dataset("therapy_val", val_dataset, "val")
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = prepare_therapy_data_tree_syn()
    print(f"\nFinal datasets:")
    print(f"  Training: {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")

