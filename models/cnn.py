import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, filters=(16, 16, 32, 32),
      kernel_sizes=(7, 3, 3, 3),
      strides=(1, 1, 2, 1)):
        super(Encoder, self).__init__()
        if len({len(filters), len(kernel_sizes), len(strides)}) != 1:
            raise ValueError(
                "length of filters/kernel_sizes/strides lists must be the same")

        convs = []
        # in_cs = [9,] + list(filters)
        in_cs = [3,] + list(filters)

        for i in range(len(filters)):
            convs.append(nn.Conv2d(in_channels=in_cs[i],
                                   out_channels=filters[i],
                                   kernel_size=kernel_sizes[i],
                                   stride=strides[i],
                                   padding=(kernel_sizes[i]-1)//2))
            convs.append(nn.ReLU(True))
            convs.append(nn.MaxPool2d(kernel_size=2))
        self.convs = nn.Sequential(*convs)

        self.bn = nn.BatchNorm2d(32)

    def forward(self, image):
        # print('en1', image.shape)
        features = self.convs(image)
        # print('en2',features.shape)
        features = self.bn(features)
        return features


