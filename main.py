from collections import deque
import gym
import numpy as np

from matplotlib import pyplot as plt
from agent import Agent
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from baselines_wrappers.monitor import Monitor
from baselines_wrappers.subproc_vec_env import SubprocVecEnv
from nn import BNN
from wrappers import make_atari_deepmind

# Defining all the required parameters
total_episodes = 500
max_steps = 100
alpha = 0.5
gamma = 1
NUM_ENVS = 4 

episodeReward = 0
totalReward = []
ep_infos = deque([], maxlen=100)

# Using the gym library to create the environment
make_env = lambda: Monitor(make_atari_deepmind('ALE/Breakout-v5'), None, allow_early_resets = True)
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)]) # Sequential
#env = SubprocVecEnv([make_env for _ in NUM_ENVS]) # For paralelism

network = BNN(alpha, env.observation_space, env.action_space.n)
agent = Agent(network, alpha, gamma)

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
                t = max_steps
                break

        states1 = states2 
        actions1 = actions2 
         
        t += 1           
              
    # Train
    print()
    print("Updating Q function ", len(data), "times")
    for state, action, reward, next_state, actions2 in data:
        agent.update(states1, actions1, rewards, states2, actions2)
    
    totalReward.append(episodeReward)

    if len(ep_infos) == 0:
        rew_mean = 0
    else: 
        rew_mean = np.mean([e['r'] for e in ep_infos])
    
    print()
    print('Episode: ', i) 
    print('Reward mean:', rew_mean)
    

env.close()

# Calculate the mean of sum of returns for each episode
meanReturn = np.mean(totalReward)

plt.plot(range(total_episodes), totalReward)
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Mean Reward per Episode')
plt.savefig('mean_reward_per_episode.png')	

# Print the results
print(f"Expected Sarsa Average Sum of Return: {meanReturn}")
