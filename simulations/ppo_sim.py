import gymnasium as gym
import numpy as np
import sys
import os
import torch
import random
import tqdm
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.ppo_agent import PPOAgent
from config.ppo_config import config

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Hyperparameters
learning_rate = config["learning_rate"]
discount_factor = config["gamma"]
hidden_size = config["hidden_size"]
num_runs = config["num_runs"]
epsilon = config["epsilon"]
total_steps = 1_000_000  # Total environment steps to run

# Seed
RANDOM_SEED = config["seed"]

# Create progress bar for runs
run_progress = tqdm.tqdm(range(num_runs), desc="Runs", position=0)

# Initialize lists to store rewards for each run
all_rewards_per_run = []
smoothed_rewards_per_run = []
step_checkpoints_per_run = []

for run in run_progress:
    run_progress.set_description(f"Run {run + 1}/{num_runs}")
    
    # Set random seed for each run
    seed = run + RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the CartPole Environment with the Record Statistics Wrapper to save results
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    
    # Create the PPO Agent
    agent = PPOAgent(
        env=env,
        state_size=4,
        hidden_size=hidden_size,
        action_size=2,
        learning_rate=learning_rate,
        gamma=discount_factor,
        epsilon=epsilon,
        device=device
    )
    
    # Initialize counters
    total_env_steps = 0
    
    # To track rewards over time
    all_rewards = []
    smoothed_rewards = []
    step_checkpoints = []
    checkpoint_interval = 1000
    next_checkpoint = checkpoint_interval

    # Create progress bar for steps
    step_progress = tqdm.tqdm(total=total_steps, desc="Steps", position=1, leave=False)
    
    # Store trajectory data for PPO updates
    trajectory = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "next_states": [],
        "dones": []
    }

    while total_env_steps < total_steps:
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select an action using the policy network
            action, log_prob, value = agent.select_action(state)
            
            # Take the action and observe the next state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store trajectory data
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["log_probs"].append(log_prob)
            trajectory["rewards"].append(reward)
            trajectory["next_states"].append(next_state)
            trajectory["dones"].append(done)
            
            # Update episode reward and total steps
            episode_reward += reward
            total_env_steps += 1
            step_progress.update(1)
            
            # If we've passed the next checkpoint, record the average reward
            if total_env_steps >= next_checkpoint:
                recent_rewards = all_rewards[-50:] if len(all_rewards) >= 50 else all_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                smoothed_rewards.append(avg_reward)
                step_checkpoints.append(total_env_steps)
                next_checkpoint += checkpoint_interval
            
            # Update state
            state = next_state
        
        # Update the policy network based on the trajectory data
        agent.update(
            trajectory["states"],
            trajectory["actions"],
            trajectory["log_probs"],
            trajectory["rewards"],
            trajectory["next_states"],
            trajectory["dones"]
        )
        
        # Clear trajectory data
        trajectory = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }
        
        # Save the total reward of this episode
        all_rewards.append(episode_reward)
    
    # Close the step progress bar at the end of each run
    step_progress.close()
    
    # Store the rewards for this run
    all_rewards_per_run.append(all_rewards)
    smoothed_rewards_per_run.append(smoothed_rewards)
    step_checkpoints_per_run.append(step_checkpoints)

# Save results to JSON file
results = {
    "step_checkpoints_per_run": step_checkpoints_per_run,
    "smoothed_rewards_per_run": smoothed_rewards_per_run,
    "all_rewards_per_run": all_rewards_per_run,
    "total_steps": total_steps,
    "num_runs": num_runs,
    "learning_rate": learning_rate,
    "discount_factor": discount_factor,
    "hidden_size": hidden_size,
    "epsilon": epsilon
}

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save to JSON file
with open("results/ppo_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results/ppo_results.json")