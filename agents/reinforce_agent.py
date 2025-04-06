import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import reinforce_config 

# Defin the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        state = nn.functional.relu(self.fc1(state))
        action_probs = nn.functional.softmax(self.fc2(state), dim=1)
        return action_probs

class ReinforceAgent:
    def __init__(
        self,
        env: gym.Env, 
        state_size=4,
        hidden_size=reinforce_config.hidden_size,  
        action_size=2, 
        learning_rate=reinforce_config.learning_rate, 
        discount_factor=reinforce_config.discount_factor,
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
        
        # Get the action probabilities from the policy network
        action_probs = self.policy_network(state)
        
        # Sample an action from the action probabilities
        action = torch.multinomial(action_probs, num_samples=1).item()

        return action, action_probs[0][action].item()
    
    def compute_returns(self, rewards):
        returns = []
        G_t = 0
        
        for reward in reversed(rewards):
            G_t = reward + self.gamma * G_t
            returns.insert(0, G_t)
        
        return returns
    
    def update_policy(self, states, actions, returns, action_probs):
        # Convert the states, actions, and returns to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Compute the log-probabilities of the taken actions
        action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs)
        
        # Compute the policy gradient
        policy_gradient = -torch.mean(log_probs * returns)
        
        # Update the policy network
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()
        
    def train(self, num_episodes):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            states, actions, rewards, action_probs = [], [], [], []

            done = False

            while not done:
                # Select an action using the policy network
                action, action_prob = self.select_action(state)
                
                # Store the state, action, reward, and action probability
                states.append(state)
                actions.append(action)
                action_probs.append(action_prob)
    
                # Take the action and observe the next state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Store the reward to calculate the return at the end of the episode
                rewards.append(reward)
                
                # Check if the episode is done
                done = terminated or truncated
                
                # Update the state
                state = next_state 
            
            # Compute the returns
            returns = self.compute_returns(rewards)
            
            # Update the policy network based on the returns
            self.update_policy(states, actions, returns, action_probs)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")