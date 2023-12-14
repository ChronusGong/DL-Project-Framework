import torch
import torch.nn as nn


# DRSN_CW Block
class RSBU_CW(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.down_sample = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride != 1:
            self.down_sample = True
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=stride)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            zero_padding=torch.zeros(input.shape).to(device)
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result
