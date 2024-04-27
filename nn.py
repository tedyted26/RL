import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class BNN(nn.Module):

    def __init__(self, lr, n_observations, n_actions):
        super(BNN, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_observations, 128),
            nn.Linear(128, 128),
            nn.Linear(128, n_actions)
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
        return self.net(x)