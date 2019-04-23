import torch.nn as nn
from torchsummary import summary

from munglinker.models.munglinker_network import MungLinkerNetwork


class BaseConvnetGlobalPooling(MungLinkerNetwork):
    """ Basic ConvNet with binary classifier sigmoid (no bias) at the end.
        The network does not flatten the convolutional features, but uses
        global max pooling to arrive at a single neuron for the output.
    """

    def __init__(self, n_input_channels=3, batch_size=4):
        super(BaseConvnetGlobalPooling, self).__init__(batch_size)

        self.n_input_channels = n_input_channels

        kernel_sizes = [3, 3, 3, 3, 3]
        paddings = [1, 1, 1, 1, 1]
        strides = [1, 1, 1, 1, 1]
        n_filters = [16, 32, 64, 64, 64]
        global_pooling_kernel = [8, 16]

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
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 32 x 128 x 256
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 64 x 64 x 128
        convRelu(2)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 128 x 32 x 64
        convRelu(3)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(2, 2))  # 128 x 16 x 32
        convRelu(4)
        cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d(2, 2))  # 128 x 8 x 16
        final_conv = nn.Conv2d(n_filters[4], 1, kernel_size=1, padding=0)
        cnn.add_module('final_conv', final_conv)

        self.cnn = cnn
        self.global_pooling = nn.AvgPool2d(global_pooling_kernel)
        self.output_activation = nn.Sigmoid()

    def forward(self, input_patch):
        conv_output = self.cnn(input_patch)
        # fcn_output = F.max_pool2d(conv_output, conv_output.size()[2:])
        fcn_output = self.global_pooling(conv_output)
        fcn_output = fcn_output.view(-1, 1)  # Flatten
        output = self.output_activation(fcn_output)
        return output


if __name__ == '__main__':
    patch_shape = 3, 256, 512
    patch_channels = 3

    model = BaseConvnetGlobalPooling(n_input_channels=patch_channels)
    print(model)
    summary(model, patch_shape, device="cpu")
