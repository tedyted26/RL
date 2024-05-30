# https://www.youtube.com/watch?v=tsy1mgB7hB0
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
from matplotlib import pyplot as plt

from baselines_wrappers.dummy_vec_env import DummyVecEnv
from baselines_wrappers.monitor import Monitor
from wrappers import BatchedPytorchFrameStack, PytorchLazyFrames, make_atari_deepmind

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ = 1000
NUM_ENVS=4

def nature_cnn(observation_space, depths=[32, 64, 64], final_layer=512):
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size = 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size = 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size = 3, stride=1),
        nn.ReLU(),
        nn.Flatten()
    )

    # Compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.num_actions = env.action_space.n

        conv_net = nature_cnn(env.observation_space)

        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        q_values = self(obses_t)
        max_q_indeces = torch.argmax(q_values, dim=1)
        actions = max_q_indeces.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                    actions[i] = random.randint(0, self.num_actions - 1)

        return actions
    
    def compute_loss(self, transitions, target_net):
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]

        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])
        else:
            obses = np.asarray(obses)
            new_obses = np.asarray(new_obses)
        
        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute Targets
        targets_online_q_values = self(new_obses_t)
        targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)
        
        targets_target_q_values = target_net(new_obses_t)
        targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=targets_online_best_q_indices)

        targets = rews_t + GAMMA * (1 - dones_t) * targets_selected_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss


make_env = lambda: Monitor(make_atari_deepmind('Tennis-v0'), None, allow_early_resets=True)
vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])

env = BatchedPytorchFrameStack(vec_env, k=4)

max_steps = 100000
n_experiments = 1
LOGGING_FREQ = 5000

all_rewards = []
all_steps = []

for ex in range(n_experiments):
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    ep_infos = deque([], maxlen=100)

    evaluation_rewards = []
    training_steps = []
    lenghts_episode = []

    episode_count = 0

    online_net = Network(env)
    target_net = Network(env)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

    # Initialize replay buffer
    obses, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

        new_obses, rews, dones, infos = env.step(actions)

        for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

        obses = new_obses

    # Main Training Loop
    obses, _ = env.reset()
    for step in range(max_steps):

        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        rnd_sample = random.random()
        
        if isinstance(obses[0], PytorchLazyFrames):
            act_obses = np.stack([o.get_frames() for o in obses])
            actions = online_net.act(act_obses, epsilon)
        else:
            actions = online_net.act(obses, epsilon)

        new_obses, rews, dones, infos = env.step(actions)

        for obs, action, rew, done, new_ob, info in zip(obses, actions, rews, dones, new_obses, infos):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

            if done:
                ep_infos.append(info['episode'])
                episode_count +=1

        obses = new_obses

        # Start gradient step
        transitions = random.sample(replay_buffer, BATCH_SIZE)

        # Compute loss
        loss = online_net.compute_loss(transitions, target_net)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Net
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

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

plt.savefig('results_ddqn_tennis_100k_x1.png')
