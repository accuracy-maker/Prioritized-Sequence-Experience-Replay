import numpy as np
import torch
# class ReplyBuffer():
#     def __init__(self,max_size,input_shape,n_acts):
#         self.mem_size = max_size
#         self.mem_cntr = 0
#         self.state_memory = np.zeros((self.mem_size,*input_shape))
#         self.new_state_memory = np.zeros((self.mem_size,*input_shape))
#         self.action_memory = np.zeros((self.mem_size, n_acts))
#         self.reward_memory = np.zeros(self.mem_size)
#         self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool_)
        
#     def store_transition(self,state,action,reward,state_,done):
#         index = self.mem_cntr % self.mem_size
#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward
#         self.terminal_memory[index] = done
        
#         self.mem_cntr += 1
        
#     def sample_buffer(self,batch_size):
#         max_mem = min(self.mem_cntr,self.mem_size)
        
#         batch = np.random.choice(max_mem,batch_size)
#         states = self.state_memory[batch]
#         states_ = self.new_state_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         dones = self.terminal_memory[batch]
        
#         return states, actions, rewards, states_, dones

# class ReplayBuffer():
#     def __init__(self, max_size, input_shape, n_acts):
#         self.mem_size = max_size
#         self.mem_cntr = 0
#         self.state_memory = np.zeros((self.mem_size, *input_shape))
#         self.new_state_memory = np.zeros((self.mem_size, *input_shape))
#         self.action_memory = np.zeros((self.mem_size, n_acts))
#         self.reward_memory = np.zeros(self.mem_size)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

#     def add(self, state, action, reward, state_, done):
#         index = self.mem_cntr % self.mem_size
#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward
#         self.terminal_memory[index] = done

#         self.mem_cntr += 1

#     def sample(self, batch_size):
#         max_mem = min(self.mem_cntr, self.mem_size)
#         batch_indices = np.random.choice(max_mem, batch_size, replace=False)

#         states = self.state_memory[batch_indices]
#         states_ = self.new_state_memory[batch_indices]
#         actions = self.action_memory[batch_indices]
#         rewards = self.reward_memory[batch_indices]
#         dones = self.terminal_memory[batch_indices]

#         return states, actions, rewards, states_, dones

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, device):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.uint8)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

        self.device = device
        
    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch

# Test the reply buffer
if __name__ == '__main__':
    import gymnasium as gym
    from tqdm import tqdm
       
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    action = env.action_space.sample()
    print(f'observation: {observation}')
    print(f'action: {action}')
    print(f'observation shape: {env.observation_space.shape}')
    print(f'action space: {env.action_space.n}')

    
    memory = ReplyBuffer(max_size=1000,input_shape=env.observation_space.shape,n_acts=env.action_space.n)
    for i in tqdm(range(100)):
        # print(f'Episode: {i+1}')
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory.store_transition(obs, action, reward, next_obs, done)
            obs = next_obs
    
        
    batch_size = 8
    states, actions, rewards, states_, dones = memory.sample_buffer(batch_size=batch_size)
    print(states)
    print(actions)
    print(rewards)
    print(states_)
    print(dones)