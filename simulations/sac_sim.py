import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gymnasium as gym
import numpy as np
import sys
import torch
import random
import tqdm
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.sac_agent import DiscreteSACAgent
from config.sac_config import config

# Set up device
device = torch.device("cpu")  #cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create progress bar for runs
run_progress = tqdm.tqdm(range(config["num_runs"]), desc="Runs", position=0)

# Initialize lists to store rewards for each run
all_rewards_per_run = []
smoothed_rewards_per_run = []
step_checkpoints_per_run = []

for run in run_progress:
    run_progress.set_description(f"Run {run + 1}/{config['num_runs']}")
    
    # Set random seed
    seed = run + config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the environment
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    
    # Create the SAC Agent
    agent = DiscreteSACAgent(
        env=env,
        state_size=4,
        hidden_size=config["hidden_size"],
        action_size=2,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        buffer_size=config["buffer_size"],
        tau=config["tau"],
        target_entropy=config["target_entropy"],
        device=device
    )
    
    # Initialize tracking variables
    total_steps = 0
    all_rewards = []
    smoothed_rewards = []
    step_checkpoints = []
    checkpoint_interval = 1000
    next_checkpoint = checkpoint_interval

    # Create progress bar for steps
    step_progress = tqdm.tqdm(total=config["total_steps"], desc="Steps", position=1, leave=False)

    while total_steps < config["total_steps"]:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            if total_steps < config["min_steps_before_learning"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            agent.buffer.push(state, action, reward, next_state, done)
            
            # Update networks if past initial collection period
            if total_steps >= config["min_steps_before_learning"]:
                for _ in range(config["updates_per_step"]):
                    agent.update(config["batch_size"])
            
            # Update tracking
            episode_reward += reward
            total_steps += 1
            step_progress.update(1)
            
            if total_steps >= next_checkpoint:
                recent_rewards = all_rewards[-50:] if len(all_rewards) >= 50 else all_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                smoothed_rewards.append(avg_reward)
                step_checkpoints.append(total_steps)
                next_checkpoint += checkpoint_interval
            
            state = next_state
            
            if done:
                all_rewards.append(episode_reward)
                break
    
    step_progress.close()
    
    # Store results for this run
    all_rewards_per_run.append(all_rewards)
    smoothed_rewards_per_run.append(smoothed_rewards)
    step_checkpoints_per_run.append(step_checkpoints)

# Save results
results = {
    "step_checkpoints_per_run": step_checkpoints_per_run,
    "smoothed_rewards_per_run": smoothed_rewards_per_run,
    "all_rewards_per_run": all_rewards_per_run,
    "config": config
}

os.makedirs("results", exist_ok=True)
with open("results/sac_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results/sac_results.json")