import random
import numpy as np
import torch as T
import torch.nn.functional as F

from nn import BNN

class Agent():
	def __init__(self, num_state, num_actions, action_space, alpha, gamma=0.99, epsilon=1.0, epsilon_dec=1e-5, epsilon_min=0.01):
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
		self.num_state = num_state
		self.num_actions = num_actions
		self.action_space = action_space
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_dec = epsilon_dec
		self.epsilon_min = epsilon_min

		self.Q = BNN(self.alpha, self.num_state, self.num_actions)
		
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
		self.Q.optimizer.zero_grad()
		prev_states = T.as_tensor(prev_state, dtype = T.float32).to(self.Q.device)
		prev_actions = T.as_tensor(prev_action, dtype=T.int64).unsqueeze(-1).to(self.Q.device)
		rewards = T.as_tensor(reward).unsqueeze(-1).to(self.Q.device)
		next_states = T.as_tensor(next_state, dtype = T.float).to(self.Q.device)
		next_actions = T.as_tensor(next_action).unsqueeze(-1).to(self.Q.device)

		# Q(S,A) for all 4 env, all actions
		q_pred_all = self.Q(prev_states)
		# Q(S,A) for all 4 env, only prev_actions
		q_pred = T.gather(q_pred_all, dim=1, index=prev_actions)

		# Q(S_,A_) for all 4 env, all actions
		q_next_all = self.Q(next_states)

		expected_q_next_all = []
		for environment in range(len(q_next_all)):
			expected_q_next = 0
			for a in range(len(q_next_all[environment])): 
				# change this line to be pi(a_) * q(S_,a_)
				expected_q_next += self.epsilon * q_next_all[environment][a]
			expected_q_next_all.append(expected_q_next)
		# Expected Q value based on the current policy (for now just epsilon greedy)
		
			# best_next_action = T.argmax(q_next_all[environment], dim=0, keepdim=True)
			# best_q_next = q_next_all[environment].gather(0, best_next_action).squeeze(-1)
			# expected_q_next = self.epsilon * q_next_all[environment].mean(dim=0) + (1 - self.epsilon) * best_q_next


		q_target = rewards + self.gamma * T.stack(expected_q_next_all) 
		
		loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
		loss.backward()
		self.Q.optimizer.step()
		self.decrement_epsilon()

	
	def choose_action(self, observations): 
		actions = []
		observations_t = T.as_tensor(observations, dtype = T.float32)
		with T.no_grad():
			action_probs = self.Q(observations_t) # Q values

			max_q_indices = T.argmax(action_probs, dim=1)
			actions = max_q_indices.detach().tolist()
			# action = T.multinomial(action_probs, 1).squeeze()

			for i in range(len(actions)):
				rnd_sample = random.random()
				if rnd_sample <= self.epsilon:
					actions[i] = random.randint(0, self.num_actions - 1)
		
		return actions
	
	def decrement_epsilon(self):
		self.epsilon = self.epsilon - self.epsilon_dec \
						if self.epsilon > self.epsilon_min else self.epsilon_min
		
		