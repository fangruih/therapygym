import torch
import argparse
import os
from pprint import pprint

def check_thinking_tags(trajectory_path):
    """Check if thinking tags are present in the conversation history."""
    print(f"\nLoading trajectory from: {trajectory_path}")
    
    # Add safe globals for deserialization
    from torch.serialization import add_safe_globals
    add_safe_globals([
        'rllm.agents.agent.Trajectory',
        'rllm.agents.agent.Step',
        'rllm.agents.agent.Action'
    ])
    
    # Load the trajectory
    trajectories = torch.load(trajectory_path, weights_only=False)
    print(f"\nNumber of trajectories: {len(trajectories)}")
    
    # Check each trajectory
    for traj_idx, trajectory in enumerate(trajectories):
        print(f"\nTrajectory {traj_idx + 1}:")
        print("=" * 40)
        
        # Check each step
        for step_idx, step in enumerate(trajectory.steps):
            print(f"\nStep {step_idx + 1}:")
            print("-" * 30)
            
            # Check messages in chat completions
            for msg in step.chat_completions:
                if msg['role'] == 'assistant':  # These are therapist responses
                    content = msg['content']
                    if "<think>" in content or "</think>" in content:
                        print("WARNING: Found thinking tags in therapist message!")
                        print(f"Message preview: {content[:200]}...")
                    else:
                        print("Therapist message is clean (no thinking tags)")
                        print(f"Message preview: {content[:200]}...")
                elif msg['role'] == 'user':  # These are patient messages
                    content = msg['content']
                    print(f"Patient message preview: {content[:200]}...")
            
            # Check if thinking tags appear in other fields
            if hasattr(step, 'model_response') and step.model_response:
                if "<think>" in step.model_response or "</think>" in step.model_response:
                    print("\nNOTE: Thinking tags found in model_response (this is expected)")
            
            if hasattr(step, 'action') and step.action:
                if isinstance(step.action, dict):
                    action_text = step.action.get('therapist_response', '')
                else:
                    action_text = str(step.action)
                if "<think>" in action_text or "</think>" in action_text:
                    print("WARNING: Found thinking tags in action!")
                    print(f"Action preview: {action_text[:200]}...")
                else:
                    print("Action is clean (no thinking tags)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_dir", help="Directory containing saved .pt trajectories")
    args = parser.parse_args()
    # Get the latest trajectory file
    log_dir = args.trajectory_dir
    trajectory_files = [f for f in os.listdir(log_dir) if f.endswith('.pt')]
    latest_file = max(trajectory_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
    trajectory_path = os.path.join(log_dir, latest_file)
    
    check_thinking_tags(trajectory_path) 
