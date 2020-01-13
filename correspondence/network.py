import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import SpiralConv


class Net(torch.nn.Module):
    def __init__(self, in_channels, num_classes, indices):
        super(Net, self).__init__()

        self.fc0 = nn.Linear(in_channels, 16)
        self.conv1 = SpiralConv(16, 32, indices)
        self.conv2 = SpiralConv(32, 64, indices)
        self.conv3 = SpiralConv(64, 128, indices)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
