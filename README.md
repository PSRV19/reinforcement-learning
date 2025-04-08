# Assignment 2: REINFORCE and basic Actor-Critic Methods

In this repo we have the following agents:

- REINFORCE
- AC
- A2C

## How to run simulations

Simply run the following command

``` python
python simulations/{agent_name}_sim.py
```

file for the desired agent and the results will be saved in `\results\` directory.

Running the `plot.py` file in the utils directory creates a `results\plots\` directory (if it doesn't exist yet) under the `\results\` directory, and creates a `{agent_name}_learning_curve.png` file with a plot of returns over episodes.

## How to use the plot.py file

1. Basic usage:

    ``` python
    python utils/plot.py --results results/{agent_name}_results.json
    ```

2. With custom window size:

    ``` python
    python utils/plot.py --results results/{agent_name}_results.json --window 100
    ```
