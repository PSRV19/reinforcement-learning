import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.distributions import Categorical
from collections import deque
import numpy as np
from config import sac_config

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

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

class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.fc(state)

class DiscreteSACAgent:
    def __init__(
        self,
        env: gym.Env,
        state_size: int,
        hidden_size: int,
        action_size: int,
        learning_rate: float,
        gamma: float,
        batch_size: int,
        buffer_size: int,
        tau: float,
        target_entropy: float,
        device=None
    ):
        self.env = env
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.tau = tau
        self.target_entropy = -1 * target_entropy * np.log(action_size)
        self.device = device or torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

        # Initialize adaptive temperature tuning
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

        # Initialize networks
        self.policy = PolicyNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.q1 = QNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.q2 = QNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.q1_target = QNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.q2_target = QNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)

        # Copy weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def select_action(self, state, evaluate=False):
        """Select an action using the policy network."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy(state)
            if evaluate:
                action = torch.argmax(probs, dim=1).item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()
        return action

    def update(self, batch_size):
        """Update networks using a batch of experiences."""
        if len(self.buffer) < batch_size:
            return

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q values
        current_q1 = self.q1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.q2(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_probs = self.policy(next_states)
            next_log_probs = torch.log(next_probs + 1e-10)
            next_q1_target = self.q1_target(next_states)
            next_q2_target = self.q2_target(next_states)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # Compute soft Q-value targets
            expected_q = (next_probs * (next_q_target - self.alpha.detach() * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * expected_q

        # Compute Q-value losses
        q1_loss = F.mse_loss(current_q1, target_q.detach())
        q2_loss = F.mse_loss(current_q2, target_q.detach())

        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy network
        probs = self.policy(states)
        log_probs = torch.log(probs + 1e-10)
        q_min = torch.min(self.q1(states), self.q2(states))
        policy_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update temperature parameter alpha
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).mean())
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()

        # Update target networks with polyak averaging
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)