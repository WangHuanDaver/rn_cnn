import torch.nn as nn
import torch.nn.functional as F
from rn_cnn_net import rn_cnn_net


# *** model structure *** #
class rn_cnn(nn.Module):
    def __init__(self):
        super(rn_cnn, self).__init__()
        fc1_in = 100352
        fc2_in = 1024
        fc3_in = 64
        self.c1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.resnet1 = rn_cnn_net(32, 64)
        self.resnet2 = rn_cnn_net(64, 128)
        self.fc1 = nn.Linear(fc1_in, fc2_in)
        self.fc2 = nn.Linear(fc2_in, 2)

        self.Maxpool = nn.MaxPool2d(2)
        self.BN1 = nn.BatchNorm1d(fc1_in)
        self.BN2 = nn.BatchNorm1d(fc2_in)
        self.BN3 = nn.BatchNorm1d(fc3_in)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.Maxpool(self.c1(x))
        x = self.Maxpool(self.resnet1(x))
        x = self.Maxpool(self.resnet2(x))

        x = x.view(batch_size, -1)
        x = self.BN1(x)
        x = F.relu(self.fc1(x))
        x = self.BN2(x)
        x = self.fc2(x)
        return x
