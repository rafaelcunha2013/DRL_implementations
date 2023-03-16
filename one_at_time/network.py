import torch.nn as nn
import torch.nn.functional as F



###########################
# Neural Network
###########################
class Network(nn.Module):

    def __init__(self, n_observations, n_actions, hid_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.layer3 = nn.Linear(hid_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)