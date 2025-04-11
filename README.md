# Assignment 2: REINFORCE and basic Actor-Critic Methods

In this repo we have the following agents:

- REINFORCE
- AC
- A2C

## How to run simulations

Ensure that you are in the `\reinforcement-learning` (root) directory. Simply run the following command:

``` python
python simulations/{agent_name}_sim.py
```

File for the desired agent and the results will be saved in `\results\` directory.

Running the `plot.py` file in the utils directory creates a `results\` directory (if it doesn't exist yet), and creates a `{agent_name}_performance.png` file with a plot of smoothed rewards over environment steps.

## How to use the plot.py file

1. Basic usage:

    ``` python
    python .\utils\plot.py --algorithm {algorithm_name}
    ```

