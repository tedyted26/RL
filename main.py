import gym
import numpy as np

from matplotlib import pyplot as plt
from agent import Agent
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from baselines_wrappers.subproc_vec_env import SubprocVecEnv
from wrappers import make_atari_deepmind

# Defining all the required parameters
epsilon = 0.1
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
# make_env = lambda: make_atari_deepmind('ALE/Breakout-v5')
# env = DummyVecEnv([make_env for _ in NUM_ENVS]) # Sequential
#env = SubprocVecEnv([make_env for _ in NUM_ENVS]) # For paralelism

env = make_atari_deepmind('ALE/Breakout-v5')

n_states = int(np.prod(env.observation_space.shape))
n_actions = env.action_space.n

agent = Agent(
	n_states, 
	n_actions, env.action_space, alpha, gamma, epsilon)

for i in range(total_episodes):
    t = 0
    state1, _ = env.reset() 
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
         
        t += 1
        episodeReward += reward
        env.render()

        if terminated: 
            break
    
        if t % 10 == 0:
            print()
            print('Step:', t)
            print('Episode: ', i)
            print('Avg Rew:', np.mean(totalReward))

    totalReward.append(episodeReward)
env.close()

# Calculate the mean of sum of returns for each episode
meanReturn = np.mean(totalReward)
	

# Print the results
print(f"Expected Sarsa Average Sum of Return: {meanReturn}")
