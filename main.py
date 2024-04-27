from collections import deque
import gym
import numpy as np

from matplotlib import pyplot as plt
from agent import Agent
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from baselines_wrappers.monitor import Monitor
from baselines_wrappers.subproc_vec_env import SubprocVecEnv
from wrappers import make_atari_deepmind

# Defining all the required parameters
total_episodes = 500
max_steps = 100
alpha = 0.5
gamma = 1
NUM_ENVS = 4 
"""
	The two parameters below is used to calculate
	the reward by each algorithm
"""
episodeReward = 0
totalReward = []
ep_infos = deque([], maxlen=100)

# Using the gym library to create the environment
make_env = lambda: Monitor(make_atari_deepmind('ALE/Breakout-v5'), None, allow_early_resets = True)
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)]) # Sequential
#env = SubprocVecEnv([make_env for _ in NUM_ENVS]) # For paralelism

#env = make_atari_deepmind('ALE/Breakout-v5')

agent = Agent(
	env.observation_space, env.action_space, alpha, gamma)

for i in range(total_episodes):
  
    # Collect data
    data = []
    t = 0
    states1, _ = env.reset() 
    actions1 = agent.choose_action(states1)
    episodeReward = 0
    while t < max_steps:

        states2, rewards, dones, _, infos = env.step(actions1)  
        actions2 = agent.choose_action(states2)

        for state, action, reward, done, state2, action2, info in zip(states1, actions1, rewards, dones, states2, actions2, infos):       
            data.append((state, action, reward, state2, action2))

            if done:
                ep_infos.append(info['episode'])

        states1 = states2 
        actions1 = actions2 
         
        t += 1           
              
    # Train
    print()
    print("Updating Q function ", len(data), "times")
    for state, action, reward, next_state, actions2 in data:
        agent.update(states1, actions1, rewards, states2, actions2)
    
    totalReward.append(episodeReward)

    rew_mean = np.mean([e['r'] for e in ep_infos]) or 0
    len_mean = np.mean([e['l'] for e in ep_infos]) or 0
    print()
    print('Reward mean:', rew_mean)
    print('Episode: ', i) 

env.close()

# Calculate the mean of sum of returns for each episode
meanReturn = np.mean(totalReward)
	

# Print the results
print(f"Expected Sarsa Average Sum of Return: {meanReturn}")
