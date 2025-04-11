import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import ac_config

# Actor network: maps state to a distribution over actions
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, state):
        return self.fc(state)

# Critic network: approximates the state value function V(s)
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.fc(state)

class ACAgent:
    def __init__(
        self,
        env: gym.Env, 
        state_size: int,
        hidden_size: int,  
        action_size: int, 
        learning_rate: float, 
        discount_factor: float,
        device=None
    ):
        self.env = env
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor and Critic networks (separate networks)
        self.actor = ActorNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.critic = CriticNetwork(self.state_size, self.hidden_size).to(self.device)
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def select_action(self, state):
        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Get action probabilities from actor
        action_probs = self.actor(state)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob

    def compute_returns(self, rewards):
        returns = []
        G_t = 0
        # Compute return as the cumulative discounted reward (Monte Carlo)
        for reward in reversed(rewards):
            G_t = reward + self.gamma * G_t
            returns.insert(0, G_t)
        return returns

    def update_policy(self, states, actions, log_probs, returns):
        # Convert lists to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        returns = np.array(returns, dtype=np.float32)

        # Convert lists to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Get state-value estimates from the critic
        values = self.critic(states).squeeze()
        
        # Actor loss: using the MC return (without subtracting a baseline)
        # Note: subtracting the critic’s prediction (i.e. computing an advantage) can reduce variance,
        # but here we implement the basic AC update as requested.
        actor_loss = -torch.mean(torch.stack(log_probs) * returns)
        
        # Critic loss: mean squared error between the estimated value and the return
        critic_loss = torch.mean((returns - values) ** 2)
        
        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
