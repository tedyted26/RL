import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class BNN(nn.Module):

    def __init__(self, lr, observation_space, n_actions):
        super(BNN, self).__init__()
        n_input_channels = observation_space.shape[0]
# n_states = int(np.prod(env.observation_space.shape))
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            
            # nn.Flatten(),
            # nn.Linear(n_observations, 128),
            # nn.Linear(128, 128),
            # nn.Linear(128, n_actions)
        )
        with T.no_grad():
            n_input = self.conv(T.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.lin = nn.Sequential(
            nn.Linear(n_input, 512), 
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # #return F.softmax(x, dim=-1)
        # return self.layer3(x)
        x = self.conv(x)
        x = self.lin(x)
        return x