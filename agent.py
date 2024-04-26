import numpy as np
import torch as T

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

		# self.soft_policy = lambda q_values: torch.softmax(q_values, dim=-1)
		
	def update(self, prev_state, next_state, reward, prev_action, next_action):
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
		prev_states = T.tensor(prev_state, dtype = T.float).to(self.Q.device)
		prev_actions = T.tensor(prev_action).to(self.Q.device)
		rewards = T.tensor(reward).to(self.Q.device)
		next_states = T.tensor(next_state, dtype = T.float).to(self.Q.device)
		next_actions = T.tensor(next_action).to(self.Q.device)

		# Q(S,A) for the selected a 
		q_pred = self.Q.forward(prev_states.flatten())[prev_actions]

		# Q(S_,A_) for each action
		q_next = self.Q.forward(next_states.flatten()) # check this step

		# Expected Q value based on the current policy
		best_next_action = T.argmax(q_next, dim=0, keepdim=True)
		best_q_next = q_next.gather(0, best_next_action).squeeze(-1)
		expected_q_next = self.epsilon * q_next.mean(dim=0) + (1 - self.epsilon) * best_q_next


		q_target = reward + self.gamma * expected_q_next 
		
		loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
		loss.backward()
		self.Q.optimizer.step()
		self.decrement_epsilon()

	
	def choose_action(self, observation): 
		observation = observation.flatten()  # Flatten the observation
		if np.random.random() > self.epsilon:
			state = T.tensor([observation], dtype=T.float).to(self.Q.device)
			actions = self.Q.forward(state)
			action = T.argmax(actions).item()
		else:
			action = self.action_space.sample()
		return action

	def decrement_epsilon(self):
		self.epsilon = self.epsilon - self.epsilon_dec \
						if self.epsilon > self.epsilon_min else self.epsilon_min
		