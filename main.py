import itertools
import numpy as np

from matplotlib import pyplot as plt
from agent import Agent
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from baselines_wrappers.monitor import Monitor
from nn import BNN
from wrappers import make_atari_deepmind

# Defining all the required parameters
n_experiments = 1
total_episodes = 50000 #5000
max_steps = 1000
n_environment_interactions = 100
# 0.1 - 0.01 - 0.001
alpha = 0.1
# 0.9 - 0.95 - 0.99
gamma = 0.99
# 80 mean at episode 8500 with 0.0001
# 0.001 - 0.0001 - 0.00025
lr= 0.00025
NUM_ENVS = 4 

# Using the gym library to create the environment
make_env = lambda: Monitor(make_atari_deepmind('Breakout-v0', max_episode_steps=max_steps), None, allow_early_resets = True)
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)]) # Sequential

for ex in range(n_experiments):
    network = BNN(env.observation_space, env.action_space.n)
    agent = Agent(network, alpha, gamma, lr)

    cost = []
    ep_infos = []  
    ep_count = 0

    total_reward_matrix = np.zeros((n_experiments, total_episodes))
    total_cost_matrix = np.zeros((n_experiments, total_episodes))
    total_lenght_matrix = np.zeros((n_experiments, total_episodes))

    while ep_count < total_episodes:
        states1, _ = env.reset()
        actions1 = [1, 1, 1, 1]
        ep_done = False
        data = []

        while not ep_done:
            # Collect data    
            states2, rewards, dones, _, infos = env.step(actions1)  
            actions2 = agent.choose_action(states2)
            for state, action, reward, done, state2, action2, info in zip(states1, actions1, rewards, dones, states2, actions2, infos):       
                data.append((state, action, reward, state2, action2))
                
                if done:
                    print('-------done', ep_count)
                    ep_infos.append(info['episode'])
                    ep_done = done
                    if ep_count >= total_episodes:
                        break
                    ep_count += 1

            states1 = states2 
            actions1 = actions2          
                    
        # Train
        # Loop through data with a step size of 4
        for r in range(0, len(data), 4):
            # Extract data for each iteration
            group = data[r:r+4]
            states, actions, rewards, next_states, next_actions = zip(*group)
            
            # Call the update function with the grouped data
            c = agent.update(states, actions, rewards, next_states, next_actions)
            cost.append(c.cpu().detach().numpy())

        
        rew_mean = np.mean([e['r'] for e in ep_infos])
        print([e['r'] for e in ep_infos])
        lenght_mean = np.mean([e['l'] for e in ep_infos])

        total_reward_matrix[ex, ep_count-1] = ep_infos[-1]['r']
        total_cost_matrix[ex,ep_count-1] = np.mean(cost)
        total_lenght_matrix[ex,ep_count-1] = ep_infos[-1]['l']

        print()
        print('Experiment: ', ex+1, '/', n_experiments)
        print('Episode: ', ep_count, 'with steps: ', ep_infos[-1]['l']) 
        print('Reward mean total: ', rew_mean)   

env.close()

mean_rewards = np.mean(total_reward_matrix, axis=0)
std_rewards = np.std(total_reward_matrix, axis=0) / np.sqrt(total_reward_matrix.shape[0])

print(total_reward_matrix)

mean_costs = np.mean(total_cost_matrix, axis=0)
std_costs = np.std(total_cost_matrix, axis=0) / np.sqrt(total_cost_matrix.shape[0])

print(mean_rewards)
print()
print(std_rewards)
mean_lengths = np.mean(total_lenght_matrix, axis=0)
std_lengths = np.std(total_lenght_matrix, axis=0) / np.sqrt(total_lenght_matrix.shape[0])

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

axes[0].plot(range(total_episodes), mean_rewards, linewidth=1, label='Mean Reward')
axes[0].fill_between(range(total_episodes), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Mean Reward')
axes[0].set_title('Mean Reward per Episode')
axes[0].legend()

axes[1].plot(range(total_episodes), mean_costs, linewidth=1, label='Mean Cost')
axes[1].fill_between(range(total_episodes), mean_costs - std_costs, mean_costs + std_costs, alpha=0.2)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Mean Cost')
axes[1].set_title('Mean Cost per Episode')
axes[1].legend()

axes[2].plot(range(total_episodes), mean_lengths, linewidth=1, label='Mean Length (timesteps)')
axes[2].fill_between(range(total_episodes), mean_lengths - std_lengths, mean_lengths + std_lengths, alpha=0.2)
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Mean Length')
axes[2].set_title('Mean Length per Episode')
axes[2].legend()

plt.tight_layout()

plt.savefig('results.png')