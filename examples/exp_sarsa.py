# main.py

import gym
import numpy as np

from matplotlib import pyplot as plt
from exp_agent import ExpectedSarsaAgent

# Using the gym library to create the environment
env = gym.make('ALE/Breakout-v5')

# Defining all the required parameters
epsilon = 0.1
total_episodes = 500
max_steps = 100
alpha = 0.5
gamma = 1
"""
	The two parameters below is used to calculate
	the reward by each algorithm
"""
episodeReward = 0
totalReward = []

# Defining all the three agents
agent = ExpectedSarsaAgent(
	epsilon, alpha, gamma, env.observation_space.n, 
	env.action_space.n, env.action_space)


for _ in range(total_episodes):
    t = 0
    state1, info = env.reset() 
    action1 = agent.choose_action(state1) 
    episodeReward = 0
    while t < max_steps:

        # Getting the next state, reward, and other parameters
        state2, reward, terminated, truncated, info = env.step(action1) 

        # Choosing the next action 
        action2 = agent.choose_action(state2) 
        
        # Learning the Q-value 
        agent.update(state1, state2, reward, action1, action2) 

        state1 = state2 
        action1 = action2 
        
        # Updating the respective vaLues 
        t += 1
        episodeReward += reward
        
        # If at the end of learning process 
        if terminated: 
            break
    # Append the sum of reward at the end of the episode
    totalReward.append(episodeReward)
env.close()

# Calculate the mean of sum of returns for each episode
meanReturn = np.mean(totalReward)
	

# Print the results
print(f"Expected Sarsa Average Sum of Return: {meanReturn}")
