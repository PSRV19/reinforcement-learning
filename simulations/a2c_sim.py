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
from agents.a2c_agent import A2CAgent
from config.a2c_config import config

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Hyperparameters
learning_rate = config["learning_rate"]
discount_factor = config["gamma"]
hidden_size = config["hidden_size"]
num_episodes = config["num_episodes"]
num_runs = config["num_runs"]
RANDOM_SEED = config["seed"]

# Dictionary to store results from all runs
results = {
    "runs": [],
    "num_episodes": num_episodes,
    "num_runs": num_runs,
    "config": {
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "hidden_size": hidden_size
    }
}

# Create progress bar for runs
run_progress = tqdm.tqdm(range(num_runs), desc="Runs", position=0)

for run in run_progress:
    run_progress.set_description(f"Run {run + 1}/{num_runs}")
    
    # Set random seed for reproducibility
    seed = run + RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the CartPole environment with statistics recording
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    
    # Create the A2C Agent
    agent = A2CAgent(
        env=env,
        state_size=4,
        hidden_size=hidden_size,
        action_size=2,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        device=device
    )
    
    # Dictionary to store results for this run
    run_results = {
        "episode_returns": [],
        "episode_steps": []
    }
    
    # Create progress bar for episodes
    episode_progress = tqdm.tqdm(range(num_episodes), desc="Episodes", position=1, leave=False)
    
    for episode in episode_progress:
        state, info = env.reset()
        ep_return = 0
        ep_length = 0
        done = False
        
        while not done:
            # Agent selects an action and returns log probability and value estimate
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_length += 1
            ep_return += reward
            
            done = terminated or truncated
            
            # Update agent on every step using the TD error update
            agent.update(state, reward, next_state, done, log_prob, value)
            
            state = next_state
            episode_progress.set_description(f"Episode {episode + 1}/{num_episodes} (Return: {ep_return:.2f})")
        
        run_results["episode_returns"].append(ep_return)
        run_results["episode_steps"].append(ep_length)
    
    results["runs"].append(run_results)
    run_progress.set_description(f"Run {run + 1}/{num_runs} completed")

# Save results to JSON file
os.makedirs("results", exist_ok=True)
with open("results/a2c_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results/a2c_results.json")
