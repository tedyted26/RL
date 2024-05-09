import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

class BNN(nn.Module):

    def __init__(self, lr, observation_space, n_actions, device):
        super(BNN, self).__init__()

        self.conv = nn.Sequential(
            # nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU(),
            # nn.Flatten(),
            
            BayesianConv2d(observation_space.num_envs, 32, (8, 8), stride=4),
            nn.ReLU(),
            BayesianConv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            BayesianConv2d(64, 64, (3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            
        )
        with T.no_grad():
            n_input = self.conv(T.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.lin = nn.Sequential(
            BayesianLinear(n_input, 512), 
            nn.ReLU(),
            BayesianLinear(512, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.KLDivLoss()
        self.to(device)

    def forward(self, x):
        x = self.conv(x)
        x = self.lin(x)
        return x