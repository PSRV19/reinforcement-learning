import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the policy network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(x)
        x = self.fc2(x)
        action_probs = torch.softmax(x, dim=1)
        return action_probs

# Define the value network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Single output for state value
        
    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(x)
        value = self.fc2(x)
        return value

class A2CAgent:
    def __init__(
        self,
        env: gym.Env,
        state_size: int,
        hidden_size: int,
        action_size: int,
        learning_rate: float,
        discount_factor: float,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        device=None,
        rollout_length: int = 5  # Number of steps per rollout
    ):
        self.env = env
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.rollout_length = rollout_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create actor and critic networks
        self.policy_network = PolicyNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.value_network = ValueNetwork(self.state_size, self.hidden_size).to(self.device)
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)
    
    def select_action(self, state):
        # Convert state to tensor and pass it through the networks
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state_tensor)
        
        # Build a categorical distribution to sample the action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        # Entropy bonus encourages exploration
        entropy = action_dist.entropy()
        value = self.value_network(state_tensor)
        
        return action.item(), log_prob, value, entropy
    
    def compute_bootstrap_returns(self, rewards, next_value, dones):
        """
        Compute n-step bootstrapped returns.
        If a terminal state is reached in a rollout step, the bootstrap value is reset to 0.
        """
        R = next_value
        returns = []
        # Iterate backward over rewards and done flags
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0  # no bootstrapping if episode terminated
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns
    
    def update_policy(self, states, actions, log_probs, values, entropies, rewards, dones, next_value):
        """
        Update policy using the collected rollout.  
        Uses the bootstrapped n-step returns, computes the advantage,
        and then updates both actor and critic networks.
        """
        # Convert buffers into tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze()
        entropies = torch.stack(entropies)
        
        # Compute bootstrapped returns
        returns = self.compute_bootstrap_returns(rewards, next_value, dones)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Advantage as the difference between returns and baseline value estimates
        advantages = returns - values
        
        # Actor loss includes an entropy bonus for exploration
        actor_loss = - (log_probs * advantages.detach()).mean() - self.entropy_coef * entropies.mean()
        
        # Critic loss (mean squared error) scales using a coefficient
        critic_loss = self.value_loss_coef * (advantages ** 2).mean()
        
        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
