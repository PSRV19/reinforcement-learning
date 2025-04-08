import gymnasium as gym
import numpy as np
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.reinforce_agent import ReinforceAgent
from config.reinforce_config import config
import torch
import tqdm
import random
import json

# Set up device in case there is a GPU
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

# Seed
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
    
    # Set a random seed for each run
    seed = run + RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the CartPole Environment with the Record Statistics Wrapper to save results
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    
    # Create the REINFORCE Agent
    agent = ReinforceAgent(
        env = env,
        state_size=4, # CartPole only has 4 states: cart position, cart velocity, pole angle, pole angular velocity
        hidden_size=hidden_size,  
        action_size=2, # In CartPole you can only move left or right
        learning_rate=learning_rate, 
        discount_factor=discount_factor,
        device = device
    )
    
    # Store results for this run
    run_results = {
        "episode_returns": [],
        "episode_steps": []
    }
    
    # Create progress bar for episodes
    episode_progress = tqdm.tqdm(range(num_episodes), desc="Episodes", position=1, leave=False)
    
    for episode in episode_progress:
        state, info = env.reset()
        states, actions, rewards = [], [], []
            
        # Store this episodes results
        ep_return = 0
        ep_length = 0

        done = False
        
        while not done:
            # Select an action using the policy network
            action = agent.select_action(state)
                
            # Store the state, action, reward, and action probability
            states.append(state)
            actions.append(action)
    
            # Take the action and observe the next state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
                
            # Increase the episode length after taking the action
            ep_length += 1
                
            # Store the reward to calculate the return at the end of the episode
            rewards.append(reward)
                
            # Check if the episode is done
            done = terminated or truncated
                
            # Update the state
            state = next_state 
            
        # Compute the returns
        returns = agent.compute_returns(rewards)
        ep_return = sum(rewards)  # Sum all rewards for the episode
        
        # Store episode return and length
        run_results["episode_returns"].append(ep_return)
        run_results["episode_steps"].append(ep_length)
            
        # Update the policy network based on the returns
        agent.update_policy(states, actions, returns)
        
        # Update episode progress bar
        episode_progress.set_description(f"Episode {episode + 1}/{num_episodes} (Return: {ep_return:.2f})")
    
    # Add this run's results to the overall results
    results["runs"].append(run_results)
    
    # Update run progress bar
    run_progress.set_description(f"Run {run + 1}/{num_runs} completed")

# Save results to JSON file
os.makedirs("results", exist_ok=True)
with open("results/reinforce_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results/reinforce_results.json")
