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
n_experiments = 10
total_episodes = 50000
max_steps = 10000
alpha = 0.5
gamma = 0.9
lr= 0.01 # probar con 1e-4
NUM_ENVS = 4 

# Using the gym library to create the environment
make_env = lambda: Monitor(make_atari_deepmind('Breakout-v0', max_episode_steps=max_steps, scale_values=True), None, allow_early_resets = True)
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)]) # Sequential

for ex in range(n_experiments-1):
    network = BNN(env.observation_space, env.action_space.n)
    agent = Agent(network, alpha, gamma, lr)

    cost = []

    total_reward_matrix = np.zeros((n_experiments, total_episodes))
    total_cost_matrix = np.zeros((n_experiments, total_episodes))

    for episode_count in range(total_episodes):
        # Collect data
        data = []
        states1, _ = env.reset() 
        actions1 = agent.choose_action(states1)
        ep_infos = deque([], maxlen=100)
        t = 1
        while t < max_steps:

            states2, rewards, dones, _, infos = env.step(actions1)  
            actions2 = agent.choose_action(states2)
            for state, action, reward, done, state2, action2, info in zip(states1, actions1, rewards, dones, states2, actions2, infos):       
                data.append((state, action, reward, state2, action2))
                
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
            c = agent.update(states, actions, rewards, next_states, next_actions)
            cost.append(c.detach().numpy())

        if len(ep_infos) == 0:
            rew_mean = 0
        else: 
            rew_mean = np.sum([e['r'] for e in ep_infos])
    
        total_reward_matrix[ex, episode_count] = rew_mean
        total_cost_matrix[ex,episode_count] = np.mean(cost)

        print()
        print('Experiment: ', ex+1, '/', n_experiments)
        print('Episode: ', episode_count, 'with steps: ', t) 
        print('Reward mean last 100 steps: ', rew_mean)   

env.close()

# Plot the learning curve with shaded error bands for mean rewards
plt.figure()
mean_rewards = np.mean(total_reward_matrix, axis=0)
std_rewards = np.std(total_reward_matrix, axis=0)
plt.plot(range(total_episodes), mean_rewards, linewidth=1, label='Mean Reward')
plt.fill_between(range(total_episodes), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Mean Reward per Episode with Error Band')
plt.legend()
plt.savefig('mean_reward_per_episode_with_error.png')

# Plot the learning curve with shaded error bands for mean cost
plt.figure()
mean_costs = np.mean(total_cost_matrix, axis=0)
std_costs = np.std(total_cost_matrix, axis=0)
plt.plot(range(total_episodes), mean_costs, linewidth=1, label='Mean Cost')
plt.fill_between(range(total_episodes), mean_costs - std_costs, mean_costs + std_costs, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('Mean Cost')
plt.title('Mean Cost per Episode with Error Band')
plt.legend()
plt.savefig('mean_cost_per_episode_with_error.png')