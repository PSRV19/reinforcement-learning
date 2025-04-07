## How to run simulations
Simply run the {agent}_sim.py file for the desired agent and the results will be saved in \simulations\results directory. Running the plot.py file in the utils directory creates a \plots directory (if it doesn't exist yet) under the \simulations\results directory, and creates a {agent}_learning_curve.png file with a plot of returns over episodes.

## How to use the plot.py file
1. Basic usage
``
python utils/plot.py --results simulations/results/reinforce_results.json
``

2. With custom window size
``
python utils/plot.py --results simulations/results/reinforce_results.json --window 100
``

3. For different agents
``
python utils/plot/py --results simulations/results/A2C_results.json 
``