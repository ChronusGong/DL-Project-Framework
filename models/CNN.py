import numpy as np
from torch import nn


def calc_output_shape(input_shape, out_channels, kernel_size, stride, padding=0, dilation=1):
    return out_channels, int((input_shape + 2*padding - (dilation*(kernel_size - 1) + 1)) / stride) + 1


def create_model():
    # 定义每个样本的长度
    length = 768

    # 定义卷积层参数
    kernel_size = 5
    out_channels = 16
    stride = 2
    padding = 0

    output_shape = calc_output_shape(length, out_channels, kernel_size, stride, padding)
    output_shape = calc_output_shape(output_shape[1], output_shape[0], 2, 2, 0)

    model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=True),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=np.prod(output_shape), out_features=1, bias=True),
        # nn.Softmax(),
    )

    return model
