import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def plot_results(results_file: str, algorithm_name: str):
    """
    Plot the results from a reinforcement learning algorithm.
    
    Args:
        results_file (str): Path to the JSON results file
        algorithm_name (str): Name of the algorithm (e.g., 'reinforce', 'a2c', 'ac')
    """
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]

    # Load the results
    with open(results_file, "r") as f:
        results = json.load(f)

    # Extract data
    step_checkpoints_per_run = results["step_checkpoints_per_run"]
    smoothed_rewards_per_run = results["smoothed_rewards_per_run"]
    num_runs = results["num_runs"]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a color palette for the runs
    colors = sns.color_palette("husl", num_runs)

    # Plot each run with a different color
    for run in range(num_runs):
        steps = step_checkpoints_per_run[run]
        rewards = smoothed_rewards_per_run[run]
        ax.plot(steps, rewards, alpha=0.7, linewidth=1.5, color=colors[run], label=f'Run {run+1}')

    # Calculate and plot the average line
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

    # Calculate mean
    mean_rewards = np.mean(interpolated_rewards, axis=0)

    # Plot mean line
    ax.plot(all_steps, mean_rewards, linewidth=3, color='black', linestyle='--', label='Average')

    # Customize the plot
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Smoothed Episode Reward', fontsize=12)
    ax.set_title(f'{algorithm_name.upper()} Performance on CartPole-v1', fontsize=14)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Format x-axis to show steps in thousands
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))

    # Ensure the results directory exists
    Path("results").mkdir(exist_ok=True)

    # Save the plot
    output_file = f"results/{algorithm_name}_performance.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot reinforcement learning results')
    parser.add_argument('--algorithm', type=str, required=True,
                      help='Name of the algorithm (e.g., reinforce, a2c, ac)')
    parser.add_argument('--results', type=str, default=None,
                      help='Path to results JSON file (default: results/{algorithm}_results.json)')
    
    args = parser.parse_args()
    
    # Set default results file path if not provided
    if args.results is None:
        args.results = f"results/{args.algorithm}_results.json"
    
    # Plot the results
    plot_results(args.results, args.algorithm)

if __name__ == "__main__":
    main() 