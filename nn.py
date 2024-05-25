import torch.nn as nn
import torch as T
import torchbnn as bnn


class BNN(nn.Module):

    def __init__(self, observation_space, n_actions, prior_mu=0, prior_sigma=0.1):
        super(BNN, self).__init__()

        self.conv = nn.Sequential(                  
            bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),       
        )

        with T.no_grad():
            # Calculate the shape of the last layer
            # Take one sample from the observation space, add the batch dimention and flatten it
            sample_input = T.rand((1,) + observation_space.shape).float() 
            n_input = self.conv(sample_input).view(1, -1).shape[1]

        self.lin = nn.Sequential(
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=n_input, out_features=512),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=512, out_features=n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.lin(x)
        return x

class DQN(nn.Module):
    def __init__(self, observation_space, n_actions):
        super(BNN, self).__init__()

        self.conv = nn.Sequential(                  
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),    
        )

        with T.no_grad():
            # Calculate the shape of the last layer
            # Take one sample from the observation space, add the batch dimention and flatten it
            sample_input = T.rand((1,) + observation_space.shape).float() 
            n_input = self.conv(sample_input).view(1, -1).shape[1]

        self.lin = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.lin(x)
        return x    