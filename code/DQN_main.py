import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from replaybuffer import ReplayBuffer
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from PESR import PrioritizedSequenceReplayBuffer
from DQN import DQN
import matplotlib.pyplot as plt
import tensorboard
from torch.utils.tensorboard import SummaryWriter


def evaluate_policy(env_name, agent, episodes, epi, seed=0):
    env = gym.make(env_name)

    returns = []
    for ep in range(episodes):
        done, total_reward = False, 0
        state, _ = env.reset(seed=seed + ep)
        #vid.start_video_recorder()
        while not done:

            state, reward, terminated, truncated, _ = env.step(agent.act(state))
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    # env.play(epi)

    return np.mean(returns), np.std(returns)


def train(env_name, model, buffer, device, log_dir,timesteps=200_000, batch_size=128,
          eps_max=0.1, eps_min=0.0, test_every=5000, seed=0):
    print(f'Training on: {env_name}, Device: {device}, Seed: {seed}')

    
    writer = SummaryWriter(log_dir)
    
    env = gym.make(env_name)


    rewards_total, stds_total = [], []
    loss_count, total_loss = 0, 0

    episode = 0
    best_reward = -np.inf

    done = False
    state, _ = env.reset(seed=seed)

    for step in range(1, timesteps + 1):
        if done:
            done = False
            state, _ = env.reset(seed=seed)
            episode += 1

        eps = eps_max - (eps_max - eps_min) * step / timesteps

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = model.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        writer.add_scalar("step reward",reward,step)
        done = terminated or truncated
        buffer.add((state, action, reward, next_state, int(done)))

        state = next_state

        if step > batch_size:
            if isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(batch_size)
                loss, td_error = model.update(batch)
                
            elif isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(batch_size)
                loss, td_error = model.update(batch, weights=weights)
                buffer.update_priorities(tree_idxs, td_error.numpy())
            
            elif isinstance(buffer,PrioritizedSequenceReplayBuffer):
                batch, is_weights, batch_indices = buffer.sample(batch_size)
                loss,td_error = model.update(batch,weights=is_weights)
                buffer.update_priorities(batch_indices,td_error.numpy())
            
            else:
                raise RuntimeError("Unknown buffer")

            total_loss += loss
            loss_count += 1

            if step % test_every == 0:

                mean, std = evaluate_policy(env_name, model, episodes=10, epi=episode, seed=seed)

                print(f"Episode: {episode}, Step: {step}, Reward mean: {mean:.2f}, Reward std: {std:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps}")

                if mean > best_reward:
                    best_reward = mean
                    model.save()

                rewards_total.append(mean)
                writer.add_scalar('episode reward',mean,episode)
                stds_total.append(std)

    return np.array(rewards_total), np.array(stds_total)


def run_experiment(config, use_priority=0, n_seeds=10):
    torch.manual_seed(0)
    mean_rewards = []

    for seed in range(n_seeds):
        
        if use_priority == 2:
            buffer = PrioritizedSequenceReplayBuffer(**config["PESR"])
        
        elif use_priority == 1:
            buffer = PrioritizedReplayBuffer(**config["PER"])
        else:
            buffer = ReplayBuffer(**config["buffer"])
        model = DQN(**config["model"])

        seed_reward, seed_std = train(seed=seed, model=model, buffer=buffer, **config["train"])
        print(f'Reward: {seed_reward}, Std: {seed_std}')
        mean_rewards.append(seed_reward)
    mean_rewards = np.array(mean_rewards)
    return mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    print('State shape: ', env.observation_space.shape[0])
    print('Number of actions: ', env.action_space.n)

    config = {
        "PER": {
            "state_size": env.observation_space.shape[0],
            "action_size": 1,  # action is discrete
            "buffer_size": 100_000,
            'eps': 1e-2,
            'alpha': .7,
            'beta': .4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            },
        
        "buffer":{
            "state_size": env.observation_space.shape[0],
            "action_size": 1,  # action is discrete
            "buffer_size": 100_000,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            },
        
        "PESR": {
            "state_size": env.observation_space.shape[0],
            "action_size": 1,
            "buffer_size": 100_000,
            "sequence_length": 5,
            "decay_rate": 0.4,
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "eps": 1e-5,
            "alpha": 0.6,
            "beta": 0.4,
        },
        
        "model": {
            "state_size": env.observation_space.shape[0],
            "action_size":  env.action_space.n,
            "gamma": 0.99,
            "lr": 1e-3,
            "tau": 0.001,
            "device":'cuda' if torch.cuda.is_available() else 'cpu',
            },
        "train": {
            "env_name": "LunarLander-v2",
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "log_dir": '/Users/gaohaitao/Prioritized-Sequence-Experience-Replay/logs/PESR',
            "timesteps": 200_000,
            "batch_size": 32,
            "test_every":5000,
            "eps_max": 0.5
            }
        }

    # mean_priority_reward, std_priority_reward = run_experiment(config, use_priority=0, n_seeds=1)
    # p_mean_priority_reward, p_std_priority_reward = run_experiment(config, use_priority=1, n_seeds=1)
    ps_mean_priority_reward, ps_std_priority_reward = run_experiment(config, use_priority=2, n_seeds=1)
    # plt.plot(mean_priority_reward,label='replaybuffer')
    # plt.plot(p_mean_priority_reward,label='PER')
    # plt.legend()
    # plt.show()