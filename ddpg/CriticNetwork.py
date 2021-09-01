import numpy as np
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

HIDDEN1_UNITS = 256
HIDDEN2_UNITS = 254

# HIDDEN1_UNITS = 300
# HIDDEN2_UNITS = 600

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.n_action = 3
        self.conv1 = nn.Sequential(
            # 输入[3,48,84]
            nn.Conv2d(
                in_channels=3,  # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=5,  # 5x5的卷积核，相当于过滤器
                stride=1,  # 卷积核在图上滑动，每隔一个扫一次
                padding=2,  # 给图外边补上0
            ),
            # 经过卷积层 输出[16,48,84] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)  # 经过池化 输出[16,16,28] 传入下一个卷积
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 同上
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32,16,28] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)  # 经过池化 输出[32,4,7] 传入输出层
        )

        self.fc1 = nn.Linear(512 + self.n_action, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.out = nn.Linear(HIDDEN2_UNITS, 1)

    def forward(self, s, a):
        x = self.conv1(s)
        x = self.conv2(x)
        x2 = th.cat((x.reshape(-1, 512), a), 1)
        x2 = F.relu(self.fc1(x2), inplace=False)
        x2 = F.relu(self.fc2(x2), inplace=False)
        out = self.out(x2)
        return out
