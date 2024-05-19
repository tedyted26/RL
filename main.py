from collections import deque
import itertools
import gym
import numpy as np

from matplotlib import pyplot as plt
from agent import Agent
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from baselines_wrappers.monitor import Monitor
from nn import BNN
from wrappers import make_atari_deepmind

# Defining all the required parameters
n_experiments = 2
total_episodes = 500
max_steps = 100
alpha = 0.5
gamma = 0.9
lr= 0.01 # probar con 1e-4
NUM_ENVS = 4 

# Using the gym library to create the environment
make_env = lambda: Monitor(make_atari_deepmind('Breakout-v0', max_episode_steps=max_steps), None, allow_early_resets = True)
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)]) # Sequential

for ex in range(n_experiments):
    network = BNN(env.observation_space, env.action_space.n)
    agent = Agent(network, alpha, gamma, lr)

    episodeReward = 0
    total_cost = 0
    totalReward = []
    ep_infos = deque([], maxlen=100)
    rew_mean_arr = []

    total_reward_matrix = np.zeros((n_experiments, total_episodes))
    total_cost_matrix = np.zeros((n_experiments, total_episodes))

    for episode_count in range(total_episodes):
        # Collect data
        data = []
        states1, _ = env.reset() 
        actions1 = agent.choose_action(states1)
        episodeReward = 0
        t = 0
        while t < max_steps:

            states2, rewards, dones, _, infos = env.step(actions1)  
            actions2 = agent.choose_action(states2)
            for state, action, reward, done, state2, action2, info in zip(states1, actions1, rewards, dones, states2, actions2, infos):       
                data.append((state, action, reward, state2, action2))
                episodeReward += reward
                
                if done:
                    print('-------done')
                    ep_infos.append(info['episode'])
                    t = max_steps
                    break

            states1 = states2 
            actions1 = actions2  

            t+=1          
                
        # Train
        # Loop through data with a step size of 4
        for r in range(0, len(data), 4):
            # Extract data for each iteration
            group = data[r:r+4]
            states, actions, rewards, next_states, next_actions = zip(*group)
            
            # Call the update function with the grouped data
            total_cost += agent.update(states, actions, rewards, next_states, next_actions)
        
        totalReward.append(episodeReward)

        if len(ep_infos) == 0:
            rew_mean = 0
        else: 
            rew_mean = np.sum([e['r'] for e in ep_infos])
    
        total_reward_matrix[ex, episode_count] = rew_mean
        # total_reward_matrix[ex, episode_count] = np.mean(totalReward)
        total_cost_matrix[ex,episode_count] = total_cost

        print()
        print('Episode: ', episode_count, 'with steps: ', t) 
        print(rew_mean)
        print('Total reward mean:', np.mean(totalReward))
        rew_mean_arr.append(rew_mean)
    

env.close()

# Calculate the mean reward across all experiments for each episode
mean_rewards_per_episode = np.mean(total_reward_matrix, axis=0)
mean_cost_per_episode = np.mean(total_cost_matrix, axis=0)

# Plot the learning curve
plt.figure()
plt.plot(range(total_episodes), mean_rewards_per_episode, linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Mean Reward per Episode')
plt.savefig('mean_reward_per_episode.png')

plt.figure()
plt.plot(range(total_episodes), mean_cost_per_episode, linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Mean Cost')
plt.title('Mean Cost per Episode')
plt.savefig('mean_cost_per_episode.png')