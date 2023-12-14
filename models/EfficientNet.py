import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientNetB0(nn.Module):
    def __init__(self, input_size, output_size):
        super(EfficientNetB0, self).__init__()

        # Modify the first layer to accept 1D input
        self.conv_stem = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        # EfficientNet-B3 blocks
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv1d(16, 24, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv1d(24, 40, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv1d(40, 80, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv1d(80, 112, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv1d(112, 192, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv1d(192, 320, kernel_size=3, stride=1, padding=1),
        )

        # Global average pooling
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer
        self.fc = nn.Linear(320, output_size)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x