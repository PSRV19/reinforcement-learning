import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Actor network: maps state to a probability distribution over actions.
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

# Critic network: approximates the state value function V(s).
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
        
        # Use separate networks for actor and critic
        self.actor = ActorNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.critic = CriticNetwork(self.state_size, self.hidden_size).to(self.device)
        
        # Separate optimizers for actor and critic networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state):
        # Convert state to tensor and get actor probabilities and critic value
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_probs = self.actor(state_tensor)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        value = self.critic(state_tensor)
        return action.item(), log_prob, value

    def update(self, state, reward, next_state, done, log_prob, value):
        # Convert next_state to tensor
        # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state = np.array(next_state, dtype=np.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # If the episode has terminated, next state's value is 0.
            next_value = self.critic(next_state_tensor) if not done else torch.tensor([[0.0]]).to(self.device)
        
        # Compute the TD error (advantage)
        td_error = reward + self.gamma * next_value - value
        
        # Actor loss: using the TD error as an advantage, detach td_error for actor update stability.
        actor_loss = -log_prob * td_error.detach()
        # Critic loss: minimize the squared TD error.
        critic_loss = td_error.pow(2)
        
        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
