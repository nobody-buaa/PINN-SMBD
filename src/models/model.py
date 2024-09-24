# 本程序用于定义网络模型及数据集
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import torch
from torch.utils.data import Dataset
import torch.nn as nn


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(13, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.fc3 = nn.Linear(1500, 1500)
        self.fc4 = nn.Linear(1500, 1500)
        self.fc5 = nn.Linear(1500, 1500)
        self.fc6 = nn.Linear(1500, 12)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        x = self.activate(self.fc3(x))
        x = self.activate(self.fc4(x))
        x = self.activate(self.fc5(x))
        x = self.fc6(x)
        return x


class DS(nn.Module):
    def __init__(self):
        super(DS, self).__init__()

        # 子结构1
        self.subnet1 = nn.Sequential(
            nn.Linear(2, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 12 * 24)
        )
        # 子结构2
        self.subnet2 = nn.Sequential(
            nn.Linear(12 * 24, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x_freq = x[:, -1].unsqueeze(1).expand(-1, 12)
        x_reshaped = torch.stack([x[:, :12], x_freq], dim=-1)

        subnet1_out = self.subnet1(x_reshaped.view(-1, 2))
        subnet1_out_reshaped = subnet1_out.view(-1, 12, 12 * 24)
        subnet1_out_permuted = subnet1_out_reshaped.permute(0, 2, 1)
        subnet1_out_flatten = subnet1_out_permuted.contiguous().view(-1, 12 * 24)

        subnet2_out = self.subnet2(subnet1_out_flatten)
        subnet2_out_reshaped = subnet2_out.view(-1, 12)

        return subnet2_out_reshaped


class DiabetesDataset(Dataset):
    def __init__(self, data, label, device):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data).to(device)
        self.y_data = torch.from_numpy(label).to(device)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
