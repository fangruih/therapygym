import os
import sys
import json
from datasets import Dataset
from therapygym.rllm.data.dataset import DatasetRegistry

def prepare_therapy_data(full_small="full"):
    """Prepare therapy dataset from patient profiles."""
    
    # Load all patient profiles
    patient_ids = []
    patient_profiles = []
    
    # Read patient dataset (portable local path)
    if full_small == "full":
        dataset_path = os.path.join(os.path.dirname(__file__), "data", "patient_psi_cm_dataset.json")
    elif full_small == "small":
        # No dedicated small file tracked here; keep behavior explicit.
        raise ValueError("small dataset file is not configured for therapygym/therapy/data; use full")
    else:
        raise ValueError(f"Invalid dataset type: {full_small}")
    
    with open(dataset_path, 'r') as f:
        patient_data = json.load(f)
        for profile in patient_data:
            patient_id = profile["id"]
            patient_ids.append(patient_id)
            patient_profiles.append(profile)
    
    # Create dataset examples
    examples = []
    for patient_id in patient_ids:
        examples.append({
            "patient_id": patient_id,
            "data_source": "therapy"  # Required by RLLM framework
        })
    
    # Split into train and validation
    train_size = int(0.8 * len(examples))
    train_examples = examples[:train_size]
    val_examples = examples[train_size:]
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("therapy_train", train_dataset, "train")
    val_dataset = DatasetRegistry.register_dataset("therapy_val", val_dataset, "val")
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = prepare_therapy_data()
    print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples") 