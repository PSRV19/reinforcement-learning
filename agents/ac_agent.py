import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs the state value
        )
        
    def forward(self, state):
        return self.fc(state)

class ACAgent:
    def __init__(self, env, state_size, hidden_size, action_size, learning_rate, discount_factor, device=None):
        self.env = env
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define separate networks for actor and critic
        self.actor = ActorNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.critic = CriticNetwork(self.state_size, self.hidden_size).to(self.device)
        
        # Optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state):
        # Convert state to tensor and add batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Get action probabilities from actor network
        action_probs = self.actor(state_tensor)
        # Create a categorical distribution and sample an action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def compute_returns(self, rewards):
        # Compute full-episode Monte Carlo returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def update_policy(self, states, actions, log_probs, returns):
        states = np.array(states, dtype=np.float32)

        # Convert states list and returns list to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Obtain the critic's value estimates for the states
        values = self.critic(states_tensor).squeeze()
        
        # Actor loss: using full Monte Carlo returns (without subtracting a baseline)
        actor_loss = torch.mean(torch.stack(log_probs) * returns_tensor)
        # Critic loss: Mean Squared Error between returns and predicted state values
        critic_loss = torch.mean((returns_tensor - values) ** 2)
        
        total_loss = actor_loss - critic_loss
        
        # Perform gradient descent steps for both actor and critic
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def update_with_mc(self, states, actions, log_probs, rewards):
        # Compute Monte Carlo returns
        returns = self.compute_returns(rewards)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Convert states to tensor
        states = np.array(states, dtype=np.float32)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        # Get critic's value estimates
        values = self.critic(states_tensor).squeeze()
        
        # Actor loss: without advantage, using full Monte Carlo returns
        actor_loss = torch.mean(torch.stack(log_probs) * returns_tensor.detach())
        # Critic loss: minimize the squared error between returns and values
        critic_loss = torch.mean((returns_tensor - values) ** 2)
        # Total loss
        total_loss = actor_loss - critic_loss
        # Update actor and critic
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()