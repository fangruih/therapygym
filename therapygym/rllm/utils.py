import os
import time
from collections import defaultdict
from typing import Dict

import openai
import torch

from therapygym.rllm.globals import GCP_LOCATION, GCP_PROJECT_ID, GEMINI_MODEL, OAI_RM_MODEL


def compute_pass_at_k(results):
    import hashlib
    import json

    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        task = trajectory.task

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = 1 if trajectory.reward > 0 else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)


def call_oai_rm_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = OAI_RM_MODEL,
    retry_count: int = int(1e9),
) -> list[str]:
    client = openai.OpenAI()

    backoff = 1
    retry_count = int(retry_count)

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as e:
            if "429" in str(e):
                print("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            else:
                print("Exception: ", e)
                return []

    if n == 1:
        content = response.choices[0].message.content
        return [content] if content is not None else []
    return [choice.message.content for choice in response.choices if choice.message.content is not None]


def call_gemini_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    project_id: str = GCP_PROJECT_ID,
    location: str = GCP_LOCATION,
    model_id: str = GEMINI_MODEL,
    retry_count: int = int(1e9),
) -> list[str]:
    """
    Calls a Gemini LLM on Vertex AI to generate n responses at a given temperature.

    Args:
        prompt (str): The text prompt to send to the LLM.
        system_prompt (str): System instruction or system prompt to send to the model.
        n (int): Number of responses to generate.
        temperature (float): Sampling temperature.
        project_id (str): Your GCP project ID.
        location (str): The region to use (e.g., us-central1).
        model_id (str): The specific Gemini model resource name.
        retry_count (int): Number of times to retry on rate-limit errors.

    Returns:
        List[str]: A list of response texts from the Gemini model.
    """

    # Optional dependency: only required if you actually use Gemini/Vertex.
    try:
        import vertexai  # type: ignore
        from google.cloud.aiplatform_v1beta1.types.content import SafetySetting  # type: ignore
        from vertexai.generative_models import (  # type: ignore
            GenerationConfig,
            GenerativeModel,
            HarmBlockThreshold,
            HarmCategory,
        )
    except Exception as e:
        raise ImportError(
            "Gemini/VertexAI support requires extra dependencies. Install: "
            "`pip install google-cloud-aiplatform vertexai` (or your org's recommended packages)."
        ) from e

    # Initialize the Vertex AI environment
    vertexai.init(project=project_id, location=location)

    # Define which harm categories to allow (or set thresholds).
    HARM_CATEGORIES = [
        HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    ]

    # Instantiate the GenerativeModel
    model = GenerativeModel(
        model_name=model_id,
        system_instruction=[system_prompt],
    )

    # Add an exponential backoff for rate limit errors
    backoff = 1
    retry_count = int(retry_count)
    generation_config = GenerationConfig(
        temperature=temperature,
        candidate_count=n,
    )

    for attempt in range(retry_count):
        try:
            # Request multiple candidates by specifying n (candidate_count)
            response = model.generate_content([prompt], generation_config=generation_config, safety_settings=[SafetySetting(category=h, threshold=HarmBlockThreshold.BLOCK_NONE) for h in HARM_CATEGORIES])
            # Once successful, break out of the retry loop
            break
        except Exception as e:
            # Retry if there's a rate-limit error (HTTP 429)
            if "429" in str(e):
                print("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            elif "403" in str(e):
                print("NO ACCESS TO ENDPOINT", e)
                raise NotImplementedError from None
            else:
                print("Exception: ", e)
                return []  # or raise an exception if desired

    # Collect the texts from all returned candidates
    # Depending on the library version, this might need to be adjusted
    # if the `response` shape is different

    try:
        # Keep this to check for errors in indexing.
        [candidate.text for candidate in response.candidates]
        if len(response.candidates) == 1:
            return response.candidates[0].text
        return [candidate.text for candidate in response.candidates]
    except Exception as e:
        print("Error extracting text from response:", e)
        return []


class RAG:
    def __init__(self, docs: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            docs (List[str]): A list of documents to encode.
            model (str): The SentenceTransformer model to use.
        """
        # Optional dependency: only required if you use RAG utilities.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise ImportError("RAG utilities require `sentence-transformers`. Install with: `pip install sentence-transformers`.") from e

        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model)
        self.docs = docs
        # Compute embeddings
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def top_k(self, query, k=1):
        # Optional dependency: only required if you use RAG utilities.
        try:
            from sentence_transformers import util  # type: ignore
        except Exception as e:
            raise ImportError("RAG utilities require `sentence-transformers`. Install with: `pip install sentence-transformers`.") from e

        # Create embedding for the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity [1 x N]
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Extract top_k indices
        top_results = torch.topk(cos_scores, k=k)

        # Prepare a list of (score, problem_text)
        results = []
        for score, idx in zip(top_results.values, top_results.indices, strict=False):
            results.append(
                {
                    "score": score,
                    "text": self.docs[int(idx)],
                    "idx": int(idx),
                }
            )
        return results


def get_log_dir():
    """Get the consistent log directory for the project.
    
    Returns:
        str: Absolute path to the log directory
    """
    # Portable default: keep logs inside the local `therapygym/therapygym` tree.
    base_dir = os.path.dirname(os.path.realpath(__file__))  # .../therapygym/therapygym/rllm
    project_dir = os.path.dirname(base_dir)  # .../therapygym/therapygym
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def save_trajectories(results, save_dir="./trajectories", filename="trajectories.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(results, save_path)
    print(f"Trajectories saved to {save_path}")
    return save_path


def collect_therapy_scores(results, is_training=False):
    """
    Collect therapy scores from trajectories.
    
    Args:
        results: List of trajectories or single trajectory
        is_training: Whether this is from training (multiple episodes)
    
    Returns:
        dict: Collected scores with structure:
            {
                'overall': float or list[float],
                'cbt_scores': dict[str, float or list[float]],
                'safety_scores': dict[str, float or list[float]],
                'evidence': dict[str, dict]  # Last evidence only
            }
    """
    import numpy as np
    from collections import defaultdict
    
    # Convert single trajectory to list
    if not isinstance(results, list):
        results = [results]
        
    # Collect scores
    overall_scores = []
    cbt_scores = defaultdict(list)
    safety_scores = defaultdict(list)
    evidence = defaultdict(list)
    
    for result in results:
        # Get info from last step
        info = result.steps[-1].info if result.steps else {}
        
        # Collect scores
        overall_scores.append(result.reward)
        
        if isinstance(info.get('cbt_scores'), dict):
            for dim, score in info['cbt_scores'].items():
                cbt_scores[dim].append(float(score))
                
        if isinstance(info.get('safety_scores'), dict):
            for dim, score in info['safety_scores'].items():
                safety_scores[dim].append(float(score))
                
        if isinstance(info.get('evidence'), dict):
            for dim, evid in info['evidence'].items():
                evidence[dim].append(evid)
    
    # Prepare scores dict
    if is_training:
        # Keep all scores as lists for training
        scores_dict = {
            'overall': overall_scores,
            'cbt_scores': {dim: scores for dim, scores in cbt_scores.items()},
            'safety_scores': {dim: scores for dim, scores in safety_scores.items()},
            'evidence': {dim: evidence[dim][-1] for dim in evidence}
        }
    else:
        # Average scores for inference
        scores_dict = {
            'overall': float(np.mean(overall_scores)) if overall_scores else 0.0,
            'cbt_scores': {dim: float(np.mean(scores)) for dim, scores in cbt_scores.items()},
            'safety_scores': {dim: float(np.mean(scores)) for dim, scores in safety_scores.items()},
            'evidence': {dim: evidence[dim][-1] for dim in evidence}
        }
    
    return scores_dict

def print_therapy_scores(scores_dict: Dict, show_evidence: bool = False):
    """Print therapy scores in a readable format."""
    print("\nScores Summary:")
    
    # Print overall score (use aggregated 'overall' key)
    overall = scores_dict.get('overall', 0.0)
    print(f"Overall Score: {overall:.2f}\n")
    
    # Print CBT dimension scores
    print("CBT Dimension Scores:")
    cbt_scores = scores_dict.get('cbt_scores', {})
    evidence = scores_dict.get('evidence', {})
    for dim, score in cbt_scores.items():
        print(f"{dim}: {score:.2f}")
        if show_evidence and dim in evidence:
            print(f"  Evidence: {evidence[dim]}")
    
    # Print safety dimension scores
    print("\nSafety Dimension Scores:")
    safety_scores = scores_dict.get('safety_scores', {})
    for dim, score in safety_scores.items():
        print(f"{dim}: {score:.2f}")
        if show_evidence and dim in evidence:
            print(f"  Evidence: {evidence[dim]}")

def store_therapy_scores(scores, save_dir="./scores", filename_prefix="therapy_scores", is_training=False):
    """
    Store and plot therapy scores including CBT and safety dimension scores.
    
    Args:
        scores (dict): Dictionary containing:
            - 'cbt_scores': dict of CBT dimension scores
            - 'safety_scores': dict of safety dimension scores
        save_dir (str): Directory to save plots and scores
        filename_prefix (str): Prefix for saved files
        is_training (bool): Whether this is from training (multiple episodes) or inference
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw scores
    scores_path = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to {scores_path}")
    
    # Create plots
    if is_training:
        # Plot CBT and safety dimensions in a single plot
        plt.figure(figsize=(15, 6))
        
        # Plot CBT dimensions
        for dim, values in scores['cbt_scores'].items():
            plt.plot(values, label=f'CBT: {dim}', linestyle='-')
        
        # Plot safety dimensions with different line style
        for dim, values in scores['safety_scores'].items():
            plt.plot(values, label=f'Safety: {dim}', linestyle='--')
        
        plt.title('Training Progress - CBT and Safety Dimensions')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_training_{timestamp}.png"))
        plt.close()
    
    else:
        # Create a single bar plot for both CBT and safety dimensions
        plt.figure(figsize=(15, 6))
        
        # Prepare data
        cbt_dims = list(scores['cbt_scores'].keys())
        cbt_values = list(scores['cbt_scores'].values())
        safety_dims = list(scores['safety_scores'].keys())
        safety_values = list(scores['safety_scores'].values())
        
        # Total number of bars and positions
        total_dims = len(cbt_dims) + len(safety_dims)
        positions = np.arange(total_dims)
        
        # Plot bars
        plt.bar(positions[:len(cbt_dims)], cbt_values, label='CBT Dimensions')
        plt.bar(positions[len(cbt_dims):], safety_values, label='Safety Dimensions')
        
        # Customize plot
        plt.title('Therapy Evaluation Scores')
        plt.xticks(positions, cbt_dims + safety_dims, rotation=45, ha='right')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_evaluation_{timestamp}.png"))
        plt.close()
