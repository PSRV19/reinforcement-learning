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

# Hyperparameters and simulation settings
learning_rate   = config["learning_rate"]
discount_factor = config["gamma"]
hidden_size     = config["hidden_size"]
num_runs        = config["num_runs"]
total_steps     = config["total_steps"]  # Total environment steps to run in each run
RANDOM_SEED     = config["seed"]

# Lists to store results from each run
all_rewards_per_run       = []  # Raw reward per episode for each run
smoothed_rewards_per_run  = []  # Smoothed (moving average) rewards sampled at checkpoints
step_checkpoints_per_run  = []  # Environment step counts at the checkpoints

# Create a progress bar for runs
run_progress = tqdm.tqdm(range(num_runs), desc="Runs", position=0)

for run in run_progress:
    run_progress.set_description(f"Run {run + 1}/{num_runs}")
    
    # Set random seed for reproducibility
    seed = run + RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the CartPole environment with episode statistics recording
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    
    # Create the A2C Agent (from a2c_agent.py)
    agent = A2CAgent(
        env=env,
        state_size=4,      # For CartPole: [position, velocity, pole angle, angular velocity]
        hidden_size=hidden_size,
        action_size=2,     # Two actions: left and right
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        device=device
    )
    
    total_env_steps = 0
    all_rewards    = []   # To record the episode reward for this run
    smoothed_rewards = []  # To store moving-average rewards at checkpoints
    step_checkpoints = []  # To record the total number of environment steps at checkpoints
    checkpoint_interval = 1000  # Record performance every 1,000 steps
    next_checkpoint = checkpoint_interval

    # Progress bar for environment steps in the current run
    step_progress = tqdm.tqdm(total=total_steps, desc="Steps", position=1, leave=False)
    
    # Run episodes until the total environment step count is reached
    while total_env_steps < total_steps:
        state, info = env.reset()
        episode_reward = 0
        done = False
        # Lists to collect data for the episode
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        # Run one episode
        while not done and total_env_steps < total_steps:
            # A2C agent selects an action; note that select_action returns (action, log_prob, value)
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transitions (for computing Monte Carlo returns later)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            episode_reward += reward
            total_env_steps += 1
            step_progress.update(1)
            
            # At checkpoints record the moving average (over the last 50 episodes, if available)
            if total_env_steps >= next_checkpoint:
                recent_rewards = all_rewards[-50:] if len(all_rewards) >= 50 else all_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                smoothed_rewards.append(avg_reward)
                step_checkpoints.append(total_env_steps)
                next_checkpoint += checkpoint_interval
                
            state = next_state
        
        # After the episode, compute MC returns and update the agent
        returns = agent.compute_returns(rewards)
        agent.update_policy(states, actions, log_probs, returns)
        
        all_rewards.append(episode_reward)
    
    step_progress.close()
    
    all_rewards_per_run.append(all_rewards)
    smoothed_rewards_per_run.append(smoothed_rewards)
    step_checkpoints_per_run.append(step_checkpoints)
    
    run_progress.set_description(f"Run {run + 1}/{num_runs} completed")

# Save the results with metadata to JSON file
results = {
    "step_checkpoints_per_run": step_checkpoints_per_run,
    "smoothed_rewards_per_run": smoothed_rewards_per_run,
    "all_rewards_per_run": all_rewards_per_run,
    "total_steps": total_steps,
    "num_runs": num_runs,
    "learning_rate": learning_rate,
    "discount_factor": discount_factor,
    "hidden_size": hidden_size
}

os.makedirs("results", exist_ok=True)
with open("results/a2c_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results/a2c_results.json")
