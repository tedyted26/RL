import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import torchbnn as bnn

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

class BNN(nn.Module):

    def __init__(self, observation_space, n_actions):
        super(BNN, self).__init__()

        self.conv = nn.Sequential(                  
            # BayesianConv2d(observation_space.shape[0], 32, (8, 8), stride=4),
            # nn.ReLU(),
            # BayesianConv2d(32, 64, (4, 4), stride=2),
            # nn.ReLU(),
            # BayesianConv2d(64, 64, (3, 3), stride=1),
            # nn.ReLU(),
            # nn.Flatten(),   
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),       
        )

        with T.no_grad():
            # Calculate the shape of the last layer
            # Take one sample from the observation space, add the batch dimention and flatten it
            sample_input = T.rand((1,) + observation_space.shape).float() 
            n_input = self.conv(sample_input).view(1, -1).shape[1]

        self.lin = nn.Sequential(
            # BayesianLinear(n_input, 512), 
            # nn.ReLU(),
            # BayesianLinear(512, n_actions)
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_input, out_features=512),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=512, out_features=n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.lin(x)
        return x