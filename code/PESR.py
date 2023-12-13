from sumtree import SumTree
import torch
import random

import numpy as np
import random
from sumtree import SumTree  

# class PrioritizedSequenceReplayBuffer:
#     def __init__(self, state_size, action_size, buffer_size, sequence_length, decay_rate, device, eps=1e-5, alpha=0.6, beta=0.4):
#         self.tree = SumTree(buffer_size)
#         self.buffer_size = buffer_size
#         self.sequence_length = sequence_length
#         self.decay_rate = decay_rate
#         self.device = device
#         self.eps = eps
#         self.alpha = alpha
#         self.beta = beta
#         self.max_priority = eps

#         # Initialize buffer components
#         self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
#         self.actions = np.zeros((buffer_size, action_size), dtype=np.int32)
#         self.rewards = np.zeros(buffer_size, dtype=np.float32)
#         self.next_states = np.zeros((buffer_size, state_size), dtype=np.float32)
#         self.dones = np.zeros(buffer_size, dtype=np.uint8)

#         self.pos = 0
#         self.size = 0

#     def _get_priority(self, error):
#         return (np.abs(error) + self.eps) ** self.alpha

#     def add(self, transition):
#         # priority = self._get_priority(error)
#         state, action, reward, next_state, done = transition

#         self._apply_decay(self.max_priority)
        
#         # Store the transition
#         self.states[self.pos] = state
#         self.actions[self.pos] = action
#         self.rewards[self.pos] = reward
#         self.next_states[self.pos] = next_state
#         self.dones[self.pos] = done

#         # Add with priority in sum tree
#         self.tree.add(self.max_priority, self.pos)

#         self.pos = (self.pos + 1) % self.buffer_size
#         self.size = min(self.size + 1, self.buffer_size)

#     def _apply_decay(self,priority):
#         for i in reversed(range(self.sequence_length - 1)):
#             idx = (self.pos - i - 1) % self.buffer_size  # Wrap around if necessary
#             tree_idx = idx + self.tree.size - 1  # Index in the tree
#             decayed_priority = priority * (self.decay_rate ** (i + 1))
#             # Ensure the priority doesn't fall below the decayed value
#             self.tree.nodes[tree_idx] = max(self.tree.nodes[tree_idx], decayed_priority)
#             self.tree.propagate(tree_idx, self.tree.nodes[tree_idx] - decayed_priority)
        
    
    
#     def sample(self,batch_size):
#         batch_indices = []
#         priorities = []
#         segment = self.tree.total / self.batch_size

#         for i in range(batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#             value = random.uniform(a, b)

#             (idx, p, data_idx) = self.tree.get(value)
#             batch_indices.append(data_idx)
#             priorities.append(p)

#         sampling_probabilities = priorities / self.tree.total
#         is_weights = np.power(self.size * sampling_probabilities, -self.beta)
#         is_weights /= is_weights.max()

#         # Get the sequences
#         states = []
#         actions = []
#         rewards = []
#         next_states = []
#         dones = []

#         for idx in batch_indices:
#             start = idx
#             end = (idx + self.sequence_length) % self.buffer_size

#             if end < start:  # Handle wraparound
#                 state_sequence = np.concatenate((self.states[start:], self.states[:end]))
#                 action_sequence = np.concatenate((self.actions[start:], self.actions[:end]))
#                 reward_sequence = np.concatenate((self.rewards[start:], self.rewards[:end]))
#                 next_state_sequence = np.concatenate((self.next_states[start:], self.next_states[:end]))
#                 done_sequence = np.concatenate((self.dones[start:], self.dones[:end]))
#             else:
#                 state_sequence = self.states[start:end]
#                 action_sequence = self.actions[start:end]
#                 reward_sequence = self.rewards[start:end]
#                 next_state_sequence = self.next_states[start:end]
#                 done_sequence = self.dones[start:end]

#             states.append(state_sequence)
#             actions.append(action_sequence)
#             rewards.append(reward_sequence)
#             next_states.append(next_state_sequence)
#             dones.append(done_sequence)

#         states = np.array(states)
#         actions = np.array(actions)
#         rewards = np.array(rewards)
#         next_states = np.array(next_states)
#         dones = np.array(dones)

#         batch = (torch.as_tensor(states).to(self.device),
#                  torch.as_tensor(actions).to(self.device),
#                  torch.as_tensor(rewards).to(self.device),
#                  torch.as_tensor(next_states).to(self.device),
#                  torch.as_tensor(dones).to(self.device))
        
#         return batch, is_weights, batch_indices

#     def update_priorities(self, batch_indices, errors):
#         for idx, error in zip(batch_indices, errors):
#             priority = self._get_priority(error)
#             self.tree.update(idx, priority)
#             self.max_priority = max(self.max_priority, priority)

class PrioritizedSequenceReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, sequence_length, decay_rate, device, eps=1e-5, alpha=0.6, beta=0.4):
        self.tree = SumTree(size=buffer_size)  # Changed to match PrioritizedReplayBuffer

        # PER params
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.

        # Initialize buffer components
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)  # Changed to match PrioritizedReplayBuffer
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)  # Changed to match PrioritizedReplayBuffer
        self.reward = torch.empty(buffer_size, dtype=torch.float)  # Changed to match PrioritizedReplayBuffer
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)  # Changed to match PrioritizedReplayBuffer
        self.done = torch.empty(buffer_size, dtype=torch.uint8)  # Changed to match PrioritizedReplayBuffer

        self.count = 0  # Changed to match PrioritizedReplayBuffer
        self.real_size = 0  # Changed to match PrioritizedReplayBuffer
        self.size = buffer_size

        # Additional variables for sequence handling
        self.sequence_length = sequence_length
        self.decay_rate = decay_rate  # Assumed decay rate for sequence priority

        # device
        self.device = device

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # Store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # Add transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # 更新计数器
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

        # # Apply decay logic only if the tree has been sufficiently populated
        # if self.real_size >= self.sequence_length and self.count >= self.sequence_length:
        #     self._apply_decay(self.max_priority)

    def _apply_decay(self, priority):

        # Apply decay directly without recalculating the new priority against the initial minimum priority
        for i in reversed(range(self.sequence_length)):
            idx = (self.count - i - 1) % self.size
            # tree_idx = idx + self.tree.size - 1 
            decayed_priority = priority * (self.decay_rate ** (i + 1))
            existing_priority = self.tree.get_priority(idx)
            self.tree.update(idx, max(decayed_priority,existing_priority))


    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # Sampling logic similar to PrioritizedReplayBuffer
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        # # Collecting the sequence of states, actions, rewards, next_states, and dones
        # states, actions, rewards, next_states, dones = [], [], [], [], []
        # for idx in sample_idxs:
        #     start_idx = idx
        #     end_idx = (start_idx + self.sequence_length) % self.size

        #     if end_idx < start_idx:  # Handle wraparound
        #         state_seq = torch.cat((self.state[start_idx:], self.state[:end_idx]), dim=0)
        #         action_seq = torch.cat((self.action[start_idx:], self.action[:end_idx]), dim=0)
        #         reward_seq = torch.cat((self.reward[start_idx:], self.reward[:end_idx]), dim=0)
        #         next_state_seq = torch.cat((self.next_state[start_idx:], self.next_state[:end_idx]), dim=0)
        #         done_seq = torch.cat((self.done[start_idx:], self.done[:end_idx]), dim=0)
        #     else:
        #         state_seq = self.state[start_idx:end_idx]
        #         action_seq = self.action[start_idx:end_idx]
        #         reward_seq = self.reward[start_idx:end_idx]
        #         next_state_seq = self.next_state[start_idx:end_idx]
        #         done_seq = self.done[start_idx:end_idx]

        #     states.append(state_seq)
        #     actions.append(action_seq)
        #     rewards.append(reward_seq)
        #     next_states.append(next_state_seq)
        #     dones.append(done_seq)

        # batch = (
        #     torch.stack(states).to(self.device),
        #     torch.stack(actions).to(self.device),
        #     torch.stack(rewards).to(self.device),
        #     torch.stack(next_states).to(self.device),
        #     torch.stack(dones).to(self.device)
        # )
        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )

        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            existing_priority = self.tree.get_priority(data_idx)
            priority = max(priority + self.eps, existing_priority * self.decay_rate) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

        if self.real_size >= self.sequence_length:
            self._apply_decay(priority)



# Test the PESR
if __name__ == "__main__":
    import gymnasium as gym
    from tqdm import tqdm
    
    # Parameters for the buffer and the environment
    state_size = 8  # State size for LunarLander
    action_size = 4  # Number of discrete actions in LunarLander
    buffer_size = 10000  # Size of the buffer
    sequence_length = 4  # Length of the sequence to be stored in the buffer
    batch_size = 32  # Batch size for sampling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the environment
    env = gym.make('LunarLander-v2')

    # Initialize the prioritized sequence replay buffer
    buffer = PrioritizedSequenceReplayBuffer(state_size, action_size, buffer_size, sequence_length, device)

    # Interact with the environment and store transitions
    for _ in range(1000):
        state,_ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Sample a random action
            next_state, reward, terminated,truncated, _ = env.step(action)
            done = terminated or truncated
            transition = (state, action, reward, next_state, done)
            buffer.add(transition)
            state = next_state

    # Test sampling from the buffer
    if buffer.real_size >= batch_size:
        sampled_batch, weights, tree_idxs = buffer.sample(batch_size)
        print("Sampled batch:", sampled_batch)
        print("Sampled weights:", weights)
        print("Tree indices:", tree_idxs)
    else:
        print("Not enough samples in the buffer to create a batch.")

    env.close()