import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def plot_results(results_files: dict[str, str]):
    """
    Plot the average performance of reinforcement learning algorithms.
    
    Args:
        results_files (dict[str, str]): A dictionary where keys are algorithm names (e.g., 'reinforce', 'a2c')
                                        and values are paths to the JSON results files.
    """
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Iterate over each algorithm and its results file
    for algorithm_name, results_file in results_files.items():
        # Load the results
        with open(results_file, "r") as f:
            results = json.load(f)

        # Extract data
        step_checkpoints_per_run = results["step_checkpoints_per_run"]
        smoothed_rewards_per_run = results["smoothed_rewards_per_run"]
        num_runs = results["num_runs"]

        # Calculate the average performance across runs
        max_steps = max([max(steps) for steps in step_checkpoints_per_run])
        step_interval = 10000  # Show steps in increments of 10,000
        all_steps = np.arange(0, max_steps + step_interval, step_interval)

        # Interpolate rewards for each run to common step points
        interpolated_rewards = []
        for run in range(num_runs):
            steps = step_checkpoints_per_run[run]
            rewards = smoothed_rewards_per_run[run]
            interpolated = np.interp(all_steps, steps, rewards)
            interpolated_rewards.append(interpolated)

        # Calculate mean rewards
        mean_rewards = np.mean(interpolated_rewards, axis=0)

        # Plot the average line for the algorithm
        ax.plot(all_steps, mean_rewards, linewidth=2, label=algorithm_name.upper())

    # Customize the plot
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Smoothed Episode Reward', fontsize=12)
    ax.set_title('Average Performance of RL Algorithms on CartPole-v1', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')

    # Format x-axis to show steps in thousands
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))

    # Ensure the results directory exists
    Path("results").mkdir(exist_ok=True)

    # Save the plot
    output_file = "results/learning_curves.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def main():
    results_files = {
        "reinforce": "results/reinforce_results.json",
        "a2c": "results/a2c_results.json",
        "a2c with entropy": "results/a2c_with_entropy_results.json",
        "ac": "results/ac_results.json"
        # "ppo": "results/ppo_results.json"
    }
    
    # Plot the results
    plot_results(results_files=results_files)

if __name__ == "__main__":
    main() 