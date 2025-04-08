import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from config import a2c_config

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = self.fc(state)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class A2CAgent:
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
        
        # Create the shared actor-critic network and its optimizer
        self.network = ActorCriticNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def select_action(self, state):
        # Convert state to tensor and forward pass through network
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_probs, state_value = self.network(state)
        
        # Create a categorical distribution over the action probabilities and sample an action
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        return action.item(), log_prob, state_value

    def update(self, state, reward, next_state, done, log_prob, value):
        # Convert next_state to tensor
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # If episode has terminated, next state's value is 0
            next_value = self.network(next_state)[1] if not done else torch.tensor([[0.0]]).to(self.device)
        
        # Compute TD error (advantage)
        td_error = reward + self.gamma * next_value - value
        
        # Compute losses:
        # Actor loss: negative log probability scaled by the advantage
        actor_loss = -log_prob * td_error.detach()
        # Critic loss: mean squared error of the TD error
        critic_loss = td_error.pow(2)
        loss = actor_loss + critic_loss
        
        # Backpropagation and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
