import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.n_feature = 6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(self.n_feature, self.n_feature, kernel_size=5)
        self.fc1 = nn.Linear(self.n_feature * 21 * 21, 50)
        self.fc2 = nn.Linear(50, 1000)

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.n_feature * 21 * 21)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
