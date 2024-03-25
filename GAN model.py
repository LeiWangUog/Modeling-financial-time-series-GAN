import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_size, output_channels, sequence_length):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.init_size = self.sequence_length // (2**5)  # 假设5次上采样，计算初始序列长度
        self.linear = nn.Linear(input_size, 256 * self.init_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 256, self.init_size)
        x = self.deconv(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.final_conv = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(x)
        return x