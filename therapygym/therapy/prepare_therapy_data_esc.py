import os
import sys
import json
from datasets import Dataset
from therapygym.rllm.data.dataset import DatasetRegistry

def prepare_therapy_data_esc(language="en", full_small="full"):
    """Prepare therapy dataset from ESC-Role character cards."""
    
    # Determine which character card file to use
    if language == "en":
        dataset_path = os.path.join(os.path.dirname(__file__), "data", "esc_character_cards_en.json")
    elif language == "zh":
        dataset_path = os.path.join(os.path.dirname(__file__), "data", "esc_character_cards_zh.json")
    else:
        raise ValueError(f"Invalid language: {language}. Must be 'en' or 'zh'")
    
    # Load character cards
    with open(dataset_path, 'r', encoding='utf-8') as f:
        character_cards = json.load(f)
    
    print(f"Loaded {len(character_cards)} {language.upper()} character cards from ESC-Eval")
    
    # Create dataset examples using indices as patient_id
    # Each character card index (0-330 for English, 0-323 for Chinese) becomes a patient_id
    examples = []
    for idx in range(len(character_cards)):
        examples.append({
            "patient_id": str(idx),  # Use string index as patient_id
            "data_source": "therapy"  # Required by RLLM framework
        })
    
    print(f"Created {len(examples)} training examples")
    
    # Handle full vs small dataset
    if full_small == "small":
        # Use only first 20 character cards for quick testing
        examples = examples[:20]
        print(f"Using small dataset: {len(examples)} examples")
    
    # Split into train and validation (80/20 split)
    train_size = int(0.8 * len(examples))
    train_examples = examples[:train_size]
    val_examples = examples[train_size:]
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("therapy_train", train_dataset, "train")
    val_dataset = DatasetRegistry.register_dataset("therapy_val", val_dataset, "val")
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"],
                       help="Language of character cards to use")
    parser.add_argument("--size", type=str, default="full", choices=["full", "small"],
                       help="Dataset size (full or small for testing)")
    args = parser.parse_args()
    
    train_dataset, val_dataset = prepare_therapy_data_esc(
        language=args.language,
        full_small=args.size
    )
    
    # Print sample
    print("\nSample training example:")
    print(train_dataset[0])
    print("\nSample validation example:")
    print(val_dataset[0])








