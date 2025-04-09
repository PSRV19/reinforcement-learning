import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, state):
        return self.fc(state)

class ReinforceAgent:
    def __init__(
        self,
        env: gym.Env, 
        state_size: int,
        hidden_size: int,  
        action_size: int, 
        learning_rate: float, 
        discount_factor: float,
        device = None
    ):
        
        self.env = env
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the policy network and optimizer
        self.policy_network = PolicyNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
    
    def select_action(self, state):
        # Convert the state to a tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Pass the state through the policy network to get the action probabilities
        action_probs = self.policy_network(state)
        
        # Create a categorical distribution over the action probabilities
        distribution = torch.distributions.Categorical(action_probs)
        
        # Sample an action from the distribution
        action = distribution.sample().item()

        return action
    
    def compute_returns(self, rewards):
        returns = []
        G_t = 0
        
        for reward in reversed(rewards):
            G_t = reward + self.gamma * G_t
            returns.insert(0, G_t)
        
        return returns
    
    def update_policy(self, states, actions, returns):
        # Convert lists to numpy arrays before creating tensors
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        returns = np.array(returns, dtype=np.float32)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Compute the log-probabilities of the taken actions
        action_probs = self.policy_network(states)
        
        # Create a categorical distribution over the action probabilities
        distribution = torch.distributions.Categorical(action_probs)
        
        # Compute the log-probabilities of the taken actions
        log_probs = distribution.log_prob(actions)
        
        # Compute the policy gradient
        policy_gradient = -torch.mean(log_probs * returns)
        
        # Update the policy network
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()