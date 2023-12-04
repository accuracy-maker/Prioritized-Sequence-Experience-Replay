from sumtree import SumTree
import torch
import random

import numpy as np
import random
from sumtree import SumTree  # Assuming this is your SumTree implementation

class PrioritizedSequenceReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, sequence_length, decay_rate, device, eps=1e-5, alpha=0.6, beta=0.4):
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.device = device
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

        # Initialize buffer components
        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.uint8)

        self.pos = 0
        self.size = 0

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, state, action, reward, next_state, done, error):
        priority = self._get_priority(error)

        self._apply_decay(priority)
        
        # Store the transition
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        # Add with priority in sum tree
        self.tree.add(priority, self.pos)

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def _apply_decay(self,priority):
        for i in reversed(range(self.sequence_length - 1)):
            idx = (self.pos - i - 1) % self.buffer_size  # Wrap around if necessary
            tree_idx = idx + self.tree.size - 1  # Index in the tree
            decayed_priority = priority * (self.decay_rate ** (i + 1))
            # Ensure the priority doesn't fall below the decayed value
            self.tree.nodes[tree_idx] = max(self.tree.nodes[tree_idx], decayed_priority)
            self.tree.propagate(tree_idx, self.tree.nodes[tree_idx] - decayed_priority)
        
    
    
    def sample(self,batch_size):
        batch_indices = []
        priorities = []
        segment = self.tree.total / self.batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)

            (idx, p, data_idx) = self.tree.get(value)
            batch_indices.append(data_idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total
        is_weights = np.power(self.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        # Get the sequences
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in batch_indices:
            start = idx
            end = (idx + self.sequence_length) % self.buffer_size

            if end < start:  # Handle wraparound
                state_sequence = np.concatenate((self.states[start:], self.states[:end]))
                action_sequence = np.concatenate((self.actions[start:], self.actions[:end]))
                reward_sequence = np.concatenate((self.rewards[start:], self.rewards[:end]))
                next_state_sequence = np.concatenate((self.next_states[start:], self.next_states[:end]))
                done_sequence = np.concatenate((self.dones[start:], self.dones[:end]))
            else:
                state_sequence = self.states[start:end]
                action_sequence = self.actions[start:end]
                reward_sequence = self.rewards[start:end]
                next_state_sequence = self.next_states[start:end]
                done_sequence = self.dones[start:end]

            states.append(state_sequence)
            actions.append(action_sequence)
            rewards.append(reward_sequence)
            next_states.append(next_state_sequence)
            dones.append(done_sequence)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        batch = (torch.as_tensor(states).to(self.device),
                 torch.as_tensor(actions).to(self.device),
                 torch.as_tensor(rewards).to(self.device),
                 torch.as_tensor(next_states).to(self.device),
                 torch.as_tensor(dones).to(self.device))
        
        return batch, is_weights, batch_indices

    def update_priorities(self, batch_indices, errors):
        for idx, error in zip(batch_indices, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)



# Test the PESR
if __name__ == "__main__":
    import gymnasium as gym
    from tqdm import tqdm
    
    # Create the CartPole environment
    env = gym.make('CartPole-v1')

    # Define buffer parameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    buffer_size = 1000
    sequence_length = 4
    batch_size = 32
    alpha = 0.6
    beta = 0.4
    decay_rate = 0.99

    # Initialize Prioritized Sequence Replay Buffer
    buffer = PrioritizedSequenceReplayBuffer(state_size, action_size, buffer_size, sequence_length, batch_size, decay_rate, eps=1e-5, alpha=0.6, beta=0.4)

    # Number of episodes for the test
    num_episodes = 10

    # Test loop
    for episode in range(num_episodes):
        state,_ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated,truncated, _ = env.step(action)
            done = terminated or truncated
            # Simulate an error for the priority calculation
            error = np.random.random()  # Normally this would be the TD-error

            # Add experience to the buffer
            buffer.add(state, action, reward, next_state, done, error)

            state = next_state

    # Sample from the buffer
    if episode >= sequence_length - 1:  # Ensure we have at least one full sequence
        states, actions, rewards, next_states, dones, is_weights, indices = buffer.sample()

        # Simulate updating priorities with new errors
        new_errors = np.random.random(size=batch_size)  # Normally these would be updated TD-errors
        buffer.update_priorities(indices, new_errors)

        # Here you would typically use the sampled transitions to update your model

