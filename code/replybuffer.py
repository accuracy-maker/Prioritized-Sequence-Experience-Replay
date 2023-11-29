import numpy as np

class ReplyBuffer():
    def __init__(self,max_size,input_shape,n_acts):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size,*input_shape))
        self.new_state_memory = np.zeros((self.mem_size,*input_shape))
        self.action_memory = np.zeros((self.mem_size, n_acts))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool_)
        
    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr,self.mem_size)
        
        batch = np.random.choice(max_mem,batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones
    
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