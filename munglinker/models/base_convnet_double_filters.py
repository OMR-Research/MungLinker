import logging

import numpy as np

import torch
import torch.nn as nn
from muscima.cropobject import CropObject
from torch.autograd import Variable
from typing import List

from torchsummary import summary

from munglinker.models.munglinker_network import MungLinkerNetwork


class BaseConvnetDoubleFilters(MungLinkerNetwork):
    def __init__(self,
                 n_input_channels=3,
                 batch_size=4):
        super(BaseConvnetDoubleFilters, self).__init__(batch_size)

        self.n_input_channels = n_input_channels

        kernel_sizes = [3, 3, 3, 3, 3]
        paddings = [1, 1, 1, 1, 1]
        strides = [1, 1, 1, 1, 1]
        n_filters = [16, 32, 64, 64, 64]

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
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 16 x 128 x 256
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 32 x 64 x 128
        convRelu(2)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 64 x 32 x 64
        convRelu(3)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(2, 2))  # 64 x 16 x 32
        convRelu(4)
        cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d(2, 2))  # 64 x 8 x 16

        self.cnn = cnn

        # Add fully connected layer stack
        fcn = nn.Sequential()
        fcn.add_module('fcn0',
                       nn.Linear(in_features=64 * 8 * 16,
                                 out_features=1,  # Output decision
                                 bias=False))

        self.fcn = fcn
        self.output_activation = nn.Sigmoid()

    def get_conv_output(self, input):
        conv_output = self.cnn(input)
        return conv_output

    def get_fcn_output_from_conv_output(self, conv_output):
        fcn_input = conv_output.view(-1, 64 * 8 * 16)
        fcn_output = self.fcn(fcn_input)
        return fcn_output

    def get_output_from_fcn_output(self, fcn_output):
        return self.output_activation(fcn_output)

    def forward(self, input):
        conv_output = self.get_conv_output(input)
        fcn_output = self.get_fcn_output_from_conv_output(conv_output)
        output = self.get_output_from_fcn_output(fcn_output)
        return output

    def prepare_patch_and_target(self, mungos_from: List[CropObject],
                                 mungos_to: List[CropObject],
                                 patches: np.ndarray,
                                 targets: np.ndarray,
                                 target_is_onehot: bool = False):
        """Does not do anything to patches.

        :param mungos_from: list of CropObjects corresponding to the FROM-half
            of the pairs; the length of the list is ``batch_size``.

        :param mungos_to: list of CropObjects corresponding to the TO-half
            of the pairs; the length of the list is ``batch_size``.

        :param patches: 4-D batch: ``batch_size x n_channels x patch_rows x patch_columns``

        :param targets: 1-D array of dim ``batch_size``, expected to be binary

        :param target_is_onehot: Expands targets to two-way softmax format.

        :param also_output_mungos: If set, outputs ``mungos_from, mungos_to, patches, targets``
            -- useful for evaluation, when you need at hand information about all the inputs.
        """
        if target_is_onehot and (targets.shape != (targets.shape[0], 2)):
            target_for_softmax = np.zeros((targets.shape[0], 2))
            target_for_softmax[range(targets.shape[0]), targets.astype('uint8')] = 1.0
        elif (not target_is_onehot) and (targets.ndim > 1):
            target_for_softmax = np.argmax(targets, axis=1)
        else:
            target_for_softmax = targets

        return mungos_from, mungos_to, patches, target_for_softmax

    def prepare_train(self, *args, **kwargs):
        mungos_from, mungos_to, X, y = self.prepare_patch_and_target(*args, **kwargs)
        return X, y

    def prepare_valid(self, *args, **kwargs):
        mungos_from, mungos_to, X, y = self.prepare_patch_and_target(*args, **kwargs)
        return mungos_from, mungos_to, X, y

    def prepare_test(self, *args, **kwargs):
        mungos_from, mungos_to, X, y = self.prepare_patch_and_target(*args, **kwargs)
        return mungos_from, mungos_to, X, y

    def prepare_runtime(self, *args, **kwargs):
        mungos_from, mungos_to, X, y = self.prepare_patch_and_target(*args, **kwargs)
        return mungos_from, mungos_to, X


if __name__ == '__main__':
    patch_shape = 3, 256, 512
    patch_channels = 3

    model = BaseConvnetDoubleFilters(n_input_channels=patch_channels)
    print(model)
    summary(model, patch_shape, device="cpu")
