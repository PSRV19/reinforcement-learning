import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from config import ppo_config

# Define the combined actor-critic network
class PPONetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PPONetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        # Policy network (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )
        # Value network (critic)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = self.fc(state)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value


class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        state_size: int,
        hidden_size: int,
        action_size: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        device=None,
    ):

        self.env = env
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the shared actor-critic network and optimizer
        self.network = PPONetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Entropy coefficient for exploration
        self.entropy_coef = 0.01                    ### HYPERPARAMETER
        self.entropy_decay = 0.995                  ### HYPERPARAMETER
        self.min_entropy_coef = 0.001               ### HYPERPARAMETER

    def select_action(self, state):
        # Convert the state to a tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Pass the state through the AC network to get the action probabilities and value estimates
        action_probs, state_value = self.network(state)

        # Create a categorical distribution over the action probabilities
        distribution = Categorical(action_probs)
        
        # Sample an action from the distribution
        action = distribution.sample()
        
        # Get log probability of the selected action
        log_prob = distribution.log_prob(action)

        return action.item(), log_prob.item(), state_value

    # Calculate Generalized Advantage Estimation (GAE)
    def compute_gae(self, rewards, values, next_values, dones, tau=0.95):   ### tau = HYPERPARAMETER
        # Initialize advantages tensor
        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0

        # Compute GAE backwards through time
        for t in reversed(range(len(rewards))):
            # Compute TD error
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            # Update advantage estimate
            advantages[t] = last_gae_lambda = delta + self.gamma * tau * (1 - dones[t]) * last_gae_lambda

        return advantages

    def update(self, states, actions, log_probs, rewards, next_states, dones):
        # Convert trajectory data to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute advantages and returns
        with torch.no_grad():
            # Get state values for current and next states
            _, state_values = self.network(states)
            _, next_state_values = self.network(next_states)
            # Compute GAE and returns
            advantages = self.compute_gae(rewards, state_values.squeeze(), next_state_values.squeeze(), dones)
            returns = advantages + state_values.squeeze()
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        batch_size = 64  # Size of each mini-batch ### HYPERPARAMETER
        num_epochs = 10  # Number of gradient updates per batch ### HYPERPARAMETER

        for _ in range(num_epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            
            # Process data in mini-batches
            for start_idx in range(0, len(states), batch_size):
                # Fill mini-batch with data
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get policy distribution and state values of current batch
                action_probs, current_values = self.network(batch_states)
                dist = Categorical(action_probs)
                current_log_probs = dist.log_prob(batch_actions)

                # Calculate probability ratios and PPO surrogate objectives
                ratios = torch.exp(current_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                
                # Calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (current_values.squeeze() - batch_returns).pow(2).mean()

                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Combine losses with entropy bonus
                total_loss = actor_loss + critic_loss - self.entropy_coef * entropy

                # Perform gradient update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

        # Decay entropy coefficient
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)
