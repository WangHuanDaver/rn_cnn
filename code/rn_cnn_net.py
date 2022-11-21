
import torch.nn as nn
import torch.nn.functional as F


# *** ResNet_net *** #
class rn_cnn_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(rn_cnn_net, self).__init__()
        self.c1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dot1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.CBN1 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.c1(F.relu(self.CBN1(x)))
        y = self.c2(F.relu(self.CBN1(y)))
        x = self.dot1(x)
        return F.relu(x + y)


