import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

class DQN(nn.Module):
    def __init__(self,state_dim,action_dim,gamma,tau,lr,device):
        super(DQN,self).__init__()
        self.device = device
        self.model =  nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        ).to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        
    def soft_update(self,target,source):
        for tp,sp in zip(target.parameters(),source.parameters()):
            tp.data.copy_((1-self.tau) * tp.data + self.tau * sp.data)
            
    def act(self,state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
            action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action
    
    def update(self,batch,weights=None):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.model.device)  # Assuming actions are long integers
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.model.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.model.device)
        
        # Compute Q values for current states
        Q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute Q values for next states using target model
        Q_next = self.target_model(next_states).max(dim=1)[0]
        Q_target = rewards + self.gamma * (1 - dones) * Q_next
        
        # Check the shapes of Q and Q_target for debugging purposes
        assert Q.shape == Q_target.shape, f"Shapes of Q and Q_target do not match: {Q.shape} vs {Q_target.shape}"

        # If weights are not provided, use uniform weights (i.e., no importance sampling)
        if weights is None:
            weights = torch.ones_like(Q, device=self.model.device)

        # Compute loss using Mean Squared Error and weights for importance sampling
        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((weights * (Q - Q_target) ** 2))

        # Perform gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft-update the target network
        with torch.no_grad():
            self.soft_update(self.target_model, self.model)
    
    def save(self):
        torch.save(self.model, "agent.pkl")