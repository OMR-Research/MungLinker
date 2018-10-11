"""This module implements a class that..."""
from __future__ import print_function, unicode_literals

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from munglinker.utils import n_onsets_from_midi_matrix


__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


MIDI_MAX_LEN = 256
MIDI_N_PITCHES = 128

MAX_N_SIMULTANEOUS_ONSETS = 12 + 1  # To have the "0" option

BATCH_SIZE = 4


class BaseConvnet(nn.Module):
    """Onset counter recurrent network.

    Heavily draws from:

    ``https://github.com/meijieru/crnn.pytorch/blob/ab61ff13c204a03d08b2836f39688c851f50fb12/models/crnn.py``

    It is a sequence-to-sequence model that in each step outputs a number of onsets
    and the loss is computed from their sum.

    The RNN produces in each step a class: how many onsets it thinks there
    are added from the current frame. There are at most ``MAX_N_SIMULTANEOUS_ONSETS``.
    """
    def __init__(self,
                 n_input_channels=3,
                 leaky_relu=False):
        super(BaseConvnet, self).__init__()

        self.n_input_channels = n_input_channels

        # Convolutional part
        # ------------------

        # Conv layer params, downstream through the stack
        kernel_sizes = [3, 3, 3, 3, 3]
        paddings = [1, 1, 1, 1, 1]
        strides = [1, 1, 1, 1, 1]
        n_filters = [8, 16, 32, 32, 32]

        n_pools = 5

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=True):
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
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i),
                               nn.BatchNorm2d(layer_n_output_channels))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
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

        # Add fully connected layer stack
        fcn = nn.Sequential()
        fcn.add_module('fcn0',
                       nn.Linear(in_features=32 * 8 * 16,
                                 out_features=2,  # Output decision
                                 bias=False))

        self.fcn = fcn

        # Output through softmax
        self.softmax = nn.Softmax()

    def get_conv_output(self, input):
        conv_output = self.cnn(input)
        return conv_output

    def get_fcn_output_from_conv_output(self, conv_output):
        fcn_input = conv_output.view(-1, 32 * 8 * 16)
        fcn_output = self.fcn(fcn_input)
        return fcn_output

    def get_output_from_fcn_output(self, fcn_output):
        return self.softmax(fcn_output)

    def forward(self, input):
        conv_output = self.get_conv_output(input)
        fcn_output = self.get_fcn_output_from_conv_output(conv_output)
        output = self.get_output_from_fcn_output(fcn_output)
        return output


def get_build_model():
    return BaseConvnet


##############################################################################


def prepare_patch_and_target(mungos_from, mungos_to, patches, targets,
                             target_is_onehot=True):
    """Does not do anything to patches.

    :param target_is_onehot: Expands targets to two-way softmax format.
    """
    if target_is_onehot and (targets.shape != (targets.shape[0], 2)):
        target_for_softmax = np.zeros((targets.shape[0], 2))
        target_for_softmax[range(targets.shape[0]), targets.astype('uint8')] = 1.0
    elif (not target_is_onehot) and (targets.ndim > 1):
        target_for_softmax = np.argmax(targets, axis=1)
    else:
        target_for_softmax = targets

    return patches, target_for_softmax


##############################################################################
# Now we define the usual model module interface.

def prepare_train(*args, **kwargs):
    X, y = prepare_patch_and_target(*args, **kwargs)
    return X, y


def prepare_valid(*args, **kwargs):
    X, y = prepare_patch_and_target(*args, **kwargs)
    return X, y


def prepare_test(*args, **kwargs):
    X, y = prepare_patch_and_target(*args, **kwargs)
    return X, y


def prepare_runtime(*args, **kwargs):
    X, _ = prepare_patch_and_target(*args, **kwargs)
    return X


def train_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator for training """
    from munglinker.batch_iterators import PoolIterator
    batch_iterator = PoolIterator(batch_size=batch_size,
                                  prepare=prepare_train,
                                  shuffle=True)
    return batch_iterator


def valid_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator for validation """
    from munglinker.batch_iterators import PoolIterator
    batch_iterator = PoolIterator(batch_size=batch_size,
                                  prepare=prepare_valid,
                                  shuffle=False)
    return batch_iterator


def test_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator for validation """
    from munglinker.batch_iterators import PoolIterator
    batch_iterator = PoolIterator(batch_size=batch_size,
                                  prepare=prepare_test,
                                  shuffle=False)
    return batch_iterator


def runtime_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator for runtime: discards the outputs """
    from munglinker.batch_iterators import PoolIterator
    # Change k_samples to a fixed number to log every k_samples batches. Effectively logs more often.
    batch_iterator = PoolIterator(batch_size=batch_size,
                                  prepare=prepare_runtime,
                                  shuffle=False)
    return batch_iterator


##############################################################################


if __name__ == '__main__':
    logging.info('Running model onset_counter in test mode.')

    # Init model
    # ----------

    _batch_size = 6
    patch_shape = 3, 256, 512
    patch_height = 256
    patch_width = 512
    patch_channels = 3

    model = BaseConvnet(n_input_channels=patch_channels,
                        leaky_relu=False)

    # Prepare dummy batch
    # -------------------
    from munglinker.utils import generate_munglinker_training_batch
    patches, targets = generate_munglinker_training_batch(_batch_size, patch_shape)

    print('patches shape: {}'.format(patches.shape))

    X, y = prepare_patch_and_target(patches, targets)
    from munglinker.utils import plot_batch_patches
    plot_batch_patches(patches, targets)

    X_torch = Variable(torch.from_numpy(X).float())
    y_torch = Variable(torch.from_numpy(y).float())

    print('X.shape = {}'.format(X.shape))
    print('y.shape = {}'.format(y.shape))

    y_pred = model(X_torch)

    print('y_true = {}'.format(y_torch))
    print('y_pred = {}'.format(y_pred))

