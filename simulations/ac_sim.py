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
from agents.ac_agent import ACAgent
from config.ac_config import config

# Set up the device (GPU if available)
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

run_progress = tqdm.tqdm(range(num_runs), desc="Runs", position=0)

for run in run_progress:
    run_progress.set_description(f"Run {run + 1}/{num_runs}")
    
    # Set random seeds for reproducibility
    seed = run + RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the CartPole environment with episode statistics logging
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    
    # Create the AC Agent
    agent = ACAgent(
        env=env,
        state_size=4,
        hidden_size=hidden_size,
        action_size=2,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        device=device
    )
    
    run_results = {
        "episode_returns": [],
        "episode_steps": []
    }
    
    episode_progress = tqdm.tqdm(range(num_episodes), desc="Episodes", position=1, leave=False)
    
    for episode in episode_progress:
        state, info = env.reset()
        done = False
        states = []
        actions = []
        rewards = []
        log_probs = []
        ep_return = 0
        ep_steps = 0
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            ep_return += reward
            ep_steps += 1
            done = terminated or truncated
            state = next_state
            episode_progress.set_description(f"Episode {episode + 1}/{num_episodes} (Return: {ep_return:.2f})")
        
        returns = agent.compute_returns(rewards)
        run_results["episode_returns"].append(ep_return)
        run_results["episode_steps"].append(ep_steps)
        
        # Update both actor and critic based on this episode's data
        agent.update_policy(states, actions, log_probs, returns)
    
    results["runs"].append(run_results)
    run_progress.set_description(f"Run {run + 1}/{num_runs} completed")

# Save simulation results to a JSON file
os.makedirs("results", exist_ok=True)
with open("results/ac_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results/ac_results.json")
