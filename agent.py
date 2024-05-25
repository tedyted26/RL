from math import gamma
import random
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim

class Agent():
	def __init__(self, network, alpha, gamma=0.99, lr=0.01, kl_weight=0.01):
		"""
		Constructor
		Args:
			num_state: The number of states
			num_actions: The number of actions
			action_space: To call the random action
			alfa: Learning rate
			gamma: The discount factor
		"""
		self.alpha = alpha
		self.gamma = gamma
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

		self.Q = network.to(self.device)

		self.mse_loss = nn.MSELoss()
		self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
		self.kl_weight = kl_weight
		self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
		
	def update(self, prev_state, prev_action, reward, next_state, next_action):
		"""
		Update the action value function using the Expected SARSA update.
		Q(S, A) = Q(S, A) + alpha(reward + (pi * Q(S_, A_) - Q(S, A))
		Args:
			prev_state: The previous state
			next_state: The next state
			reward: The reward for taking the respective action
			prev_action: The previous action
			next_action: The next action
		Returns:
			None
		"""
		# Turn the numpy arrays into pytorch tensors
		self.optimizer.zero_grad()
		prev_states = T.as_tensor(np.array(prev_state), dtype = T.float32).to(self.device)
		prev_actions = T.as_tensor(prev_action, dtype=T.int64).to(self.device)
		rewards = T.as_tensor(reward).to(self.device)
		next_states = T.as_tensor(np.array(next_state), dtype = T.float).to(self.device)

		# Q(S,A) for all 4 env, all actions
		q_pred_all = self.Q(prev_states)
		# Q(S,A) for all 4 env, only prev_actions
		q_pred = T.gather(q_pred_all, dim=1, index=prev_actions.unsqueeze(0))

		# Q(S_,A_) for all 4 env, all actions
		q_next_all = self.Q(next_states)
		# Probabilities
		with T.no_grad():
			actions_softmax = F.softmax(q_next_all, dim=-1)

		# Expected Q value based on probabilities
		expected_q_next_all = []
		for environment in range(len(q_next_all)):
			expected_q_next = 0
			for a in range(len(q_next_all[environment])):
				expected_q_next += actions_softmax[environment][a] * q_next_all[environment][a]
			expected_q_next_all.append(expected_q_next)

		q_target = rewards + self.gamma * T.stack(expected_q_next_all)
		
		mse_loss = self.mse_loss(q_target.unsqueeze(0), q_pred).to(self.device)
		kl_loss = self.kl_loss(self.Q).to(self.device)
		cost = mse_loss + self.kl_weight*kl_loss

		cost.backward()
		self.optimizer.step()

		return cost

	
	def choose_action(self, observations): 
		observations_t = T.as_tensor(observations, dtype = T.float32).to(self.device)
		with T.no_grad():
			action_probs = self.Q(observations_t) # Q values

			actions_softmax = F.softmax(action_probs, dim=-1)
			sampled_actions = T.multinomial(actions_softmax, 1)
		
		return sampled_actions

class AgentDQN():
	def __init__(self, network, alpha, gamma=0.99, lr=0.01, kl_weight=0.01):
		"""
		Constructor
		Args:
			num_state: The number of states
			num_actions: The number of actions
			action_space: To call the random action
			alfa: Learning rate
			gamma: The discount factor
			epsilon: The degree of exploration
		"""
		self.alpha = alpha
		self.gamma = gamma
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

		self.Q = network.to(self.device)

		self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		
	def update(self, prev_state, prev_action, reward, next_state, next_action):
		"""
		Update the action value function using the Expected SARSA update.
		Q(S, A) = Q(S, A) + alpha(reward + (pi * Q(S_, A_) - Q(S, A))
		Args:
			prev_state: The previous state
			next_state: The next state
			reward: The reward for taking the respective action
			prev_action: The previous action
			next_action: The next action
		Returns:
			None
		"""
		# Turn the numpy arrays into pytorch tensors
		self.optimizer.zero_grad()
		prev_states = T.as_tensor(np.array(prev_state), dtype = T.float32).to(self.device)
		prev_actions = T.as_tensor(prev_action, dtype=T.int64).to(self.device)
		rewards = T.as_tensor(reward).to(self.device)
		next_states = T.as_tensor(np.array(next_state), dtype = T.float).to(self.device)

		# Compute Targets
		# targets = r + gamma * target q vals * (1 - dones)
		target_q_values = target_net(new_obses_t)
		max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

		targets = rews_t + self.gamma * (1 - dones_t) * max_target_q_values

		# Compute Loss
		q_values = online_net(obses_t)
		action_q_values = T.gather(input=q_values, dim=1, index=actions_t)

		loss = nn.functional.smooth_l1_loss(action_q_values, targets)

		# Gradient Descent
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss

	
	def choose_action(self, observations): 
		observations_t = T.as_tensor(observations, dtype = T.float32).to(self.device)
		with T.no_grad():
			action_probs = self.Q(observations_t) # Q values

			actions_softmax = F.softmax(action_probs, dim=-1)
			sampled_actions = T.multinomial(actions_softmax, 1)
		
		return sampled_actions
	
		
		