import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)  # Convert to action probabilities
        return x

# Value Network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Outputs a single value (V(s))
    
    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

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
        
        # Create the actor (policy network) and critic (value network)
        self.policy_network = PolicyNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.value_network = ValueNetwork(self.state_size, self.hidden_size).to(self.device)
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)
    
    def select_action(self, state):
        # Convert the state to a tensor and add a batch dimension
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Pass the state through the policy network to get action probabilities
        action_probs = self.policy_network(state)
        
        # Sample an action from the probability distribution
        action = torch.multinomial(action_probs, 1).item()
        
        # Get the log probability of the selected action
        log_prob = torch.log(action_probs.squeeze(0)[action])
        
        # Get the value estimate for the current state
        value = self.value_network(state)
        
        return action, log_prob, value

    def compute_returns(self, rewards):
        returns = []
        G_t = 0
        for reward in reversed(rewards):
            G_t = reward + self.gamma * G_t
            returns.insert(0, G_t)
        return returns

    def update_policy(self, states, log_probs, returns):
        # Convert lists to numpy arrays before creating tensors
        states = np.array(states, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Create tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Actor loss: use returns directly (vanilla AC update)
        actor_loss = -torch.mean(torch.stack(log_probs) * returns)
        
        # Critic update: predict values for all states and compute the MSE loss with the returns
        values = self.value_network(states).squeeze()
        critic_loss = torch.mean((returns - values) ** 2)
        
        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
