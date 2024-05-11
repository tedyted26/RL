import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

class BNN(nn.Module):

    def __init__(self, lr, observation_space, n_actions):
        super(BNN, self).__init__()

        self.conv = nn.Sequential(                  
            BayesianConv2d(observation_space.shape[0], 32, (8, 8), stride=4),
            nn.ReLU(),
            BayesianConv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            BayesianConv2d(64, 64, (3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),        
        )

        with T.no_grad():
            # Calculate the shape of the last layer
            # Take one sample from the observation space, add the batch dimention and flatten it
            sample_input = T.rand((1,) + observation_space.shape).float() 
            n_input = self.conv(sample_input).view(1, -1).shape[1]

        self.lin = nn.Sequential(
            BayesianLinear(n_input, 512), 
            nn.ReLU(),
            BayesianLinear(512, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.conv(x)
        x = self.lin(x)
        return x