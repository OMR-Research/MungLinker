import torch.nn as nn
from torchsummary import summary

from munglinker.models.munglinker_network import MungLinkerNetwork


class BaseConvnet(MungLinkerNetwork):
    """Basic ConvNet with binary classifier sigmoid (no bias) at the end."""

    def __init__(self, n_input_channels=3, batch_size=4):
        super(BaseConvnet, self).__init__(batch_size)

        self.n_input_channels = n_input_channels

        kernel_sizes = [3, 3, 3, 3, 3]
        paddings = [1, 1, 1, 1, 1]
        strides = [1, 1, 1, 1, 1]
        n_filters = [8, 16, 32, 32, 32]

        cnn = nn.Sequential()

        def convRelu(i):
            """Builds the i-th convolutional layer/batch-norm block."""
            layer_n_input_channels = n_input_channels
            if i != 0:
                layer_n_input_channels = n_filters[i - 1]
            layer_n_output_channels = n_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(layer_n_input_channels,
                                     layer_n_output_channels,
                                     kernel_sizes[i],
                                     strides[i],
                                     paddings[i]))
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(layer_n_output_channels))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # Input size expected (n_batch x n_channels x MIDI_N_PITCHES x MIDI_MAX_LEN),
        # which is by default (n_batch x 3 x 256 x 512)
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 8 x 128 x 256
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 16 x 64 x 128
        convRelu(2)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 32 x 32 x 64
        convRelu(3)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(2, 2))  # 32 x 16 x 32
        convRelu(4)
        cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d(2, 2))  # 32 x 8 x 16

        self.cnn = cnn
        fcn = nn.Sequential()
        fcn.add_module('fcn0',
                       nn.Linear(in_features=32 * 8 * 16,
                                 out_features=1,
                                 bias=False))

        self.fcn = fcn
        self.output_activation = nn.Sigmoid()

    def forward(self, input):
        conv_output = self.cnn(input)
        fcn_input = conv_output.view(-1, 32 * 8 * 16)
        fcn_output = self.fcn(fcn_input)
        output = self.output_activation(fcn_output)
        return output


if __name__ == '__main__':
    patch_shape = 3, 256, 512
    patch_channels = 3

    model = BaseConvnet(n_input_channels=patch_channels)
    print(model)
    summary(model, patch_shape, device="cpu")
