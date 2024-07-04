import torch
import torch.nn as nn
import torch.nn.functional as F

class Modified_sim_network(nn.Module):
    def __init__(self):
        super(Modified_sim_network, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        # self.conv2 = torch.nn.Conv2d(1, 1, 3, padding=1)
        # self.conv3 = torch.nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, input):   # input : 1 * 441 * 441
        x = F.relu(self.conv(input))    # 1 * 441 * 441
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        return x

modified_sim_network = Modified_sim_network()
modified_sim_network = nn.DataParallel(modified_sim_network, range(1))