import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

class DQN(nn.Module):
    def __init__(self,state_size,action_size,gamma,tau,lr,device):
        super(DQN,self).__init__()
        self.device = device
        self.model =  nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
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
    
    def update(self,batch,weights=None,buffer=None):
        states, actions, rewards, next_states, dones = batch
        # states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
        # actions = torch.tensor(actions, dtype=torch.long, device=self.model.device)  # Assuming actions are long integers
        # rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
        # next_states = torch.tensor(next_states, dtype=torch.float32, device=self.model.device)
        # dones = torch.tensor(dones, dtype=torch.float32, device=self.model.device)
        
        # # Compute Q values for current states
        # Q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # # Compute Q values for next states using target model
        # Q_next = self.target_model(next_states).max(dim=1)[0]
        # Q_target = rewards + self.gamma * (1 - dones) * Q_next
        
        if buffer == None or buffer == 'PER':
            Q_next = self.target_model(next_states).max(dim=1).values
            Q_target = rewards + self.gamma * (1 - dones) * Q_next
            Q = self.model(states)[torch.arange(len(actions)), actions.to(torch.long).flatten()]
        else:
            batch_size, seq_len, state_dim = next_states.shape
            next_states_flat = next_states.view(batch_size * seq_len, state_dim)
            states_flat = states.view(batch_size * seq_len, state_dim)
            
            Q_next_flat = self.target_model(next_states_flat).max(dim=1).values
            Q_next = Q_next_flat.view(batch_size, seq_len)[:, -1]
            # print(f'Q_next shape: {Q_next.shape}')
            # print(f'reward shape: {rewards.shape}')
            # print(f'dones shape: {dones.shape}')
            last_rewards = rewards[:,-1]
            last_dones = dones[:,-1]
            Q_target = last_rewards + self.gamma * (1 - last_dones) * Q_next
            Q_flat = self.model(states_flat)
            Q_reshaped = Q_flat.view(batch_size,seq_len,-1)
            Q = Q_reshaped[:,0,:].max(dim=1).values
            # Q = Q_flat.view(batch_size, seq_len)[:, 0]
        
        
        # Check the shapes of Q and Q_target for debugging purposes
        assert Q.shape == Q_target.shape, f"Shapes of Q and Q_target do not match: {Q.shape} vs {Q_target.shape}"

        # If weights are not provided, use uniform weights (i.e., no importance sampling)
        if weights is None:
            weights = torch.ones_like(Q)

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
        
        return loss.item(), td_error
    
    def save(self):
        torch.save(self.model, "agent.pkl")