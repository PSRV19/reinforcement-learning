import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
import argparse

# Set the style for better visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_learning_curves(results_path, window_size=50):
    """
    Plot learning curves from a results JSON file.
    
    Args:
        results_path (str): Path to the results JSON file
        window_size (int): Size of the moving average window
    """
    # Load the results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract data
    num_episodes = results['num_episodes']
    num_runs = results['num_runs']
    runs = results['runs']
    
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Plot each run with moving average
    all_returns = []
    
    for i, run in enumerate(runs):
        returns = run['episode_returns']
        # Apply moving average
        smoothed_returns = uniform_filter1d(returns, size=window_size)
        all_returns.append(returns)
        ax.plot(range(1, num_episodes + 1), smoothed_returns, 
                alpha=0.3, label=f'Run {i+1}')
    
    # Calculate and plot the average with moving average
    all_returns = np.array(all_returns)
    avg_returns = np.mean(all_returns, axis=0)
    smoothed_avg = uniform_filter1d(avg_returns, size=window_size)
    ax.plot(range(1, num_episodes + 1), smoothed_avg, 
            linewidth=3, color='black', label='Average')
    
    # Add labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Return ({window_size}-episode moving average)')
    ax.set_title('Learning Curves')
    
    # Add legend
    ax.legend()
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(results_path), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save the plot
    plot_name = os.path.splitext(os.path.basename(results_path))[0] + '_learning_curves.png'
    plt.savefig(os.path.join(plots_dir, plot_name))
    plt.close()
    
    print(f"Plot saved to: {os.path.join(plots_dir, plot_name)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot learning curves from reinforcement learning results')
    parser.add_argument('--results', type=str, required=True,
                      help='Path to the results JSON file')
    parser.add_argument('--window', type=int, default=50,
                      help='Size of the moving average window (default: 50)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Plot the results
    plot_learning_curves(args.results, args.window)

if __name__ == "__main__":
    main()
