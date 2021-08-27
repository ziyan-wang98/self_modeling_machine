import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, initial_filters, output_size, output_channels=3,
               name="decoder"):
        super(Decoder, self).__init__()

        self.output_size = output_size

        filters = initial_filters
        filters_next = max(8, filters // 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=filters,
                      out_channels=filters,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=filters,
                      out_channels=filters_next,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1))

        filters = filters_next
        # filters_next = max(9, filters // 2)
        filters_next = max(3, filters // 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=filters,
                      out_channels=filters,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=filters,
                      out_channels=filters_next,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1))

        filters = filters_next
        filters_next = 3

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=filters,
                      out_channels=filters_next,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1))


        self.bn = nn.BatchNorm2d(32)

    def forward(self, features):
        N, C, H, W =  features.shape
        # print('de1', features.shape)

        features = self.conv1(features)
        # print('de2', features.shape)

        H *= 2
        W *= 2
        features = F.upsample(features, size=(H, W), mode='bilinear')
        features = self.conv2(features)
        # print('de2', features.shape)

        features = self.conv3(features)

        # print('de3', features.shape)

        return features


