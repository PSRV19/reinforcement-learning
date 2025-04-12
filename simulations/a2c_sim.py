import gymnasium as gym
import numpy as np
import sys
import os
import torch
import random
import tqdm
import json

# Add the parent directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.a2c_agent import A2CAgent
from config.a2c_config import config

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Hyperparameters and settings from config, plus total steps for simulation
learning_rate   = config["learning_rate"]
discount_factor = config["gamma"]
hidden_size     = config["hidden_size"]
num_runs        = config["num_runs"]
total_steps     = config["total_steps"]  # Total environment steps to run
RANDOM_SEED     = config["seed"]
rollout_length  = config.get("rollout_length", 5)  # New parameter: rollout length

# Lists to store results from each run
all_rewards_per_run     = []  # Per-episode returns per run
smoothed_rewards_per_run = []  # Moving averages of episode returns at checkpoints
step_checkpoints_per_run = []  # Environment steps at which checkpoints were recorded

# Progress bar for runs
run_progress = tqdm.tqdm(range(num_runs), desc="Runs", position=0)

for run in run_progress:
    run_progress.set_description(f"Run {run + 1}/{num_runs}")
    
    # Set random seeds for reproducibility
    seed = run + RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the CartPole environment with the RecordEpisodeStatistics wrapper
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    
    # Create the A2C agent
    agent = A2CAgent(
        env=env,
        state_size=4,      # For CartPole: [cart position, cart velocity, pole angle, pole angular velocity]
        hidden_size=hidden_size,
        action_size=2,     # Two possible actions: left or right
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        device=device,
        rollout_length=rollout_length
    )
    
    total_env_steps = 0
    all_rewards = []    # Rewards for each episode during this run
    smoothed_rewards = []  # Moving averages of episode returns
    step_checkpoints = []  # Step counts for checkpoints
    checkpoint_interval = 1000  # Record a checkpoint every 1000 steps
    next_checkpoint = checkpoint_interval
    
    # Buffers for storing transitions in the current rollout
    rollout_states = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_values = []
    rollout_entropies = []
    rollout_rewards = []
    rollout_dones = []
    
    state, info = env.reset()
    episode_reward = 0
    progress_bar = tqdm.tqdm(total=total_steps, desc="Steps", position=1, leave=False)
    
    while total_env_steps < total_steps:
        # Select action using the A2C agent
        action, log_prob, value, entropy = agent.select_action(state)
        
        # Execute the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store the transition in the rollout buffer
        rollout_states.append(state)
        rollout_actions.append(action)
        rollout_log_probs.append(log_prob)
        rollout_values.append(value)
        rollout_entropies.append(entropy)
        rollout_rewards.append(reward)
        rollout_dones.append(done)
        
        episode_reward += reward
        total_env_steps += 1
        progress_bar.update(1)
        
        # If the rollout buffer is full or the episode ends, perform an update.
        if len(rollout_states) >= rollout_length or done:
            if done:
                next_value = 0
            else:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                next_value = agent.value_network(next_state_tensor).item()
            agent.update_policy(
                rollout_states,
                rollout_actions,
                rollout_log_probs,
                rollout_values,
                rollout_entropies,
                rollout_rewards,
                rollout_dones,
                next_value
            )
            
            # Clear the rollout buffers for the next batch of transitions.
            rollout_states = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_values = []
            rollout_entropies = []
            rollout_rewards = []
            rollout_dones = []
        
        # If the episode finished, record reward data and reset the environment.
        if done:
            all_rewards.append(episode_reward)
            # Record a checkpoint if appropriate.
            if total_env_steps >= next_checkpoint:
                recent_rewards = all_rewards[-50:] if len(all_rewards) >= 50 else all_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                smoothed_rewards.append(avg_reward)
                step_checkpoints.append(total_env_steps)
                next_checkpoint += checkpoint_interval
            state, info = env.reset()
            episode_reward = 0
        else:
            state = next_state
    
    progress_bar.close()
    all_rewards_per_run.append(all_rewards)
    smoothed_rewards_per_run.append(smoothed_rewards)
    step_checkpoints_per_run.append(step_checkpoints)
    
    run_progress.set_description(f"Run {run + 1}/{num_runs} completed")

# Save results and configuration metadata into a JSON file
results = {
    "step_checkpoints_per_run": step_checkpoints_per_run,
    "smoothed_rewards_per_run": smoothed_rewards_per_run,
    "all_rewards_per_run": all_rewards_per_run,
    "total_steps": total_steps,
    "num_runs": num_runs,
    "learning_rate": learning_rate,
    "discount_factor": discount_factor,
    "hidden_size": hidden_size,
    "rollout_length": rollout_length
}

os.makedirs("results", exist_ok=True)
with open("results/a2c_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results/a2c_results.json")
