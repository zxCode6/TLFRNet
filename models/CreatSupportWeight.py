import torch
import torch.nn as nn
import torch.nn.functional as F

class SupportWeightNetwork(nn.Module):
    def __init__(self):
        super(SupportWeightNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 3)  # 全连接层，将输入映射到输出
        self.fc2 = nn.Linear(3, 1)
        # self.fc3 = nn.Linear(3, 3)
        # self.fc4 = nn.Linear(3, 1)

    def forward(self, x): # 75 * 5 * 441
        x = x.permute(0, 2, 1) # 75 * 441 * 5
        x = F.relu(self.fc1(x)) # 75 * 441 * 3
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return x  # 75 * 441 * 1

supportWeightModel = SupportWeightNetwork()
supportWeightModel = nn.DataParallel(supportWeightModel, range(1))