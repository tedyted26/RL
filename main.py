from collections import deque
import itertools
import numpy as np

from matplotlib import pyplot as plt
from agent import Agent
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from baselines_wrappers.monitor import Monitor
from nn import BNN
from wrappers import make_atari_deepmind

# Defining all the required parameters
max_steps = 1000
n_experiments = 1
n_environment_interactions = 100
LOGGING_FREQ = 100
env = 'Breakout'

# 0.1 - 0.01 - 0.001
alpha = 0.1
# 0.9 - 0.95 - 0.99
gamma = 0.99
# 80 mean at episode 8500 with 0.0001
# 0.001 - 0.0001 - 0.00025
lr= 0.01
NUM_ENVS = 4 


# Using the gym library to create the environment
make_env = lambda: Monitor(make_atari_deepmind(env+'-v0'), None, allow_early_resets = True)
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)]) # Sequential

for ex in range(n_experiments):
    network = BNN(env.observation_space, env.action_space.n)
    agent = Agent(network, alpha, gamma, lr)

    cost = []
    ep_infos = deque([], 100)  
    ep_count = 0
    data = []

    states1, _ = env.reset()

    if env == 'Breakout':
        actions1 = [1, 1, 1, 1] # If its the first ever action of breakout, fire the ball
    else:
        actions1 = agent.choose_action(states1)

    while steps < max_steps: 
        
        if len(data) < n_environment_interactions:
            # Collect data    
            states2, rewards, dones, _, infos = env.step(actions1) 
            steps =+ 1 
            actions2 = agent.choose_action(states2)
            for state, action, reward, done, state2, action2, info in zip(states1, actions1, rewards, dones, states2, actions2, infos):       
                data.append((state, action, reward, state2, action2))
                
                if done:
                    print('-------done', ep_count)
                    ep_infos.append(info['episode'])
                    ep_done = done
                    ep_count += 1

            states1 = states2 
            actions1 = actions2          
        else:            
            # Train
            # Loop through data with a step size of 4
            for r in range(0, len(data), 4):
                # Extract data for each iteration
                group = data[r:r+4]
                states, actions, rewards, next_states, next_actions = zip(*group)
                
                # Call the update function with the grouped data
                c = agent.update(states, actions, rewards, next_states, next_actions)
                cost.append(c.cpu().detach().numpy())

        
        # Logging
        if step % LOGGING_FREQ == 0:
            if len(ep_infos) == 0:
                rew_mean = 0
                len_mean = 0
            else:
                rew_mean = np.mean([e['r'] for e in ep_infos])
                len_mean = np.mean([e['l'] for e in ep_infos])

            evaluation_rewards.append(rew_mean)
            training_steps.append(step)

            print()
            print('Step:', step)
            print('Avg Rew:', rew_mean)
            print('Avg Ep Len', len_mean)
            print('Episodes', episode_count)
            print('Experiment: ', ex)
    
    all_rewards.append(evaluation_rewards)
    all_steps.append(training_steps)

# Plotting the results
plt.figure(figsize=(12, 8))

mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0) / np.sqrt(n_experiments)

plt.plot(training_steps, mean_rewards, linewidth=1, label='Mean Reward')
plt.fill_between(training_steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

plt.xlabel("Gradient-Descent Steps")
plt.ylabel("Evaluation-Time Total Reward")
plt.title("Evaluation-Time Total Reward vs Gradient-Descent Steps")
plt.legend()

plt.savefig('results_sarsa.png')