import gym
import numpy as np

from matplotlib import pyplot as plt
from agent import Agent
from baselines_wrappers.dummy_vec_env import DummyVecEnv
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

# Using the gym library to create the environment
make_env = lambda: make_atari_deepmind('ALE/Breakout-v5')
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)]) # Sequential
#env = SubprocVecEnv([make_env for _ in NUM_ENVS]) # For paralelism

#env = make_atari_deepmind('ALE/Breakout-v5')

n_states = int(np.prod(env.observation_space.shape))
n_actions = env.action_space.n

agent = Agent(
	n_states, n_actions, env.action_space, alpha, gamma)

for i in range(total_episodes):
  
    # Collect data
    data = []
    t = 0
    states1, _ = env.reset() 
    actions1 = agent.choose_action(states1)
    episodeReward = 0
    while t < max_steps:

        states2, rewards, _, _, infos = env.step(actions1)  
        actions2 = agent.choose_action(states2)

        for state, action, reward, state2, action2 in zip(states1, actions1, rewards, states2, actions2):       
            data.append((state, action, reward, state2, action2))

        states1 = states2 
        actions1 = actions2 
         
        t += 1
        for r in rewards:
            episodeReward += r
    
        if t % 10 == 0:
            print()
            print('Step:', t)
            print('Episode: ', i)
            print('Avg Rew:', np.mean(totalReward))
    
    # Train
    for state, action, reward, next_state, actions2 in data:
        agent.update(states1, actions1, rewards, states2, actions2)
    
    totalReward.append(episodeReward)
env.close()

# Calculate the mean of sum of returns for each episode
meanReturn = np.mean(totalReward)
	

# Print the results
print(f"Expected Sarsa Average Sum of Return: {meanReturn}")
