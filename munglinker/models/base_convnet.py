"""This module implements a class that..."""
from __future__ import print_function, unicode_literals

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from e2eomr.utils import n_onsets_from_midi_matrix


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

        self._counter_vec = Variable(_counter_vec)
        if torch.cuda.is_available():
            self._counter_vec.cuda()

        # Convolutional part
        # ------------------
        kernel_sizes = [3, 3, 3]
        paddings = [1, 1, 1]
        strides = [1, 1, 1]
        n_filters = [8, 16, 32]

        n_pools = 3

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

        # Input size expected (n_batch x 1 x MIDI_N_PITCHES x MIDI_MAX_LEN),
        # which is by default (n_batch x 1 x 128 x 256)
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 8 x 64 x 128
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 16 x 32 x 64
        convRelu(2)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 32 x 16 x 32

        self.cnn = cnn

        # Add output fully connected layer

        self.softmax = nn.Softmax2d()

    def get_conv_output(self, input):
        conv_output = self.cnn(input)
        return conv_output

    def forward(self, input):
        conv_output = self.get_conv_output(input)
        fcn_output = self.get_fcn_output_from_conv_output(conv_output)
        output = self.get_output_from_fcn_output(fcn_output)
        return output


def get_build_model():
    return BaseConvnet


##############################################################################


def prepare_patch_and_target(sheet, spec, midi, o2c):
    """Takes the batch genreated from the StaffPool and prepares it
    for the OnsetCounterRNN() model. Currently, the simplest counting
    setting is used: outputs a batch of (X, y) pairs where X are MIDI
    matrices and ``y`` are the onset counts.

    Because the o2c onset counts are derived from the sheet, we derive
    the onset count from the MIDI matrix.

    Note that we expect the data to be lists, since the data points are
    variable-length sequences.

    If a MIDI matrix has more frames than ``MIDI_MAX_LEN``, it will
    be squashed to this length (as though the music was sped up).

    :param midi: A list of 3-D midi matrices (single-channel-first).

    :return:
    """
    # Prepare output
    onset_counts = [n_onsets_from_midi_matrix(mm[0]) for mm in midi]

    # Comparing with o2c
    # print('onset counts: {}'.format(onset_counts))
    # print('o2c counts:   {}'.format([oc.shape for oc in o2c]))
    y = np.array(onset_counts)

    # Prepare input
    batch_size = len(midi)
    # print('Input MIDI shapes: {}'.format([m.shape for m in midi]))
    size_normalized_midis = []
    for m in midi:
        if m.shape[-1] > MIDI_MAX_LEN:
            import cv2
            m_squashed = cv2.resize(src=m[0],
                                    dsize=(MIDI_MAX_LEN, m.shape[1]),  # cv2 dim order
                                    interpolation=cv2.INTER_NEAREST)
            size_normalized_midis.append(m_squashed[np.newaxis, :, :])
        else:
            size_normalized_midis.append(m)

    X_batch_lengths = [m.shape[-1] for m in size_normalized_midis]
    # print('X_batch_lengths = {}'.format(X_batch_lengths))
    # print('orig midi lengths = {}'.format([m.shape[-1] for m in midi]))
    # print('resized midi lengths = {}'.format([m.shape[-1] for m in size_normalized_midis]))

    # We'll need to reshape the inputs to (n_channels, n_frames, n_bins),
    # but this will be done in the RNN part of the CRNN.
    # (Note that for sheet data, we need to set the background
    #  to white, or - rather - invert the images!)
    X_np_data = np.zeros((batch_size, 1, MIDI_N_PITCHES, MIDI_MAX_LEN))
    for i, m in enumerate(size_normalized_midis):
        X_np_data[i, 0, :, :X_batch_lengths[i]] = m

    return X_np_data, y





##############################################################################
# Now we define the usual model module interface.

def prepare_train(sheet, spec, midi, o2c):
    X, y = prepare_patch_and_target(sheet, spec, midi, o2c)
    return X, y


def prepare_valid(sheet, spec, midi, o2c):
    X, y = prepare_patch_and_target(sheet, spec, midi, o2c)
    return X, y


def prepare_test(sheet, spec, midi, o2c):
    X, y = prepare_patch_and_target(sheet, spec, midi, o2c)
    return X, y


def prepare_runtime(sheet, spec, midi, o2c):
    X, _ = prepare_patch_and_target(sheet, spec, midi, o2c)
    return X


def train_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator for training """
    from msmd.data_pools.batch_iterators import VariableLengthSequencePoolIterator
    from msmd.data_pools.batch_iterators import MultiviewPoolIteratorUnsupervised
    batch_iterator = VariableLengthSequencePoolIterator(batch_size=batch_size,
                                                        prepare=prepare_train,
                                                        shuffle=True)
    return batch_iterator


def valid_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator for validation """
    from msmd.data_pools.batch_iterators import VariableLengthSequencePoolIterator
    batch_iterator = VariableLengthSequencePoolIterator(batch_size=batch_size,
                                                        prepare=prepare_valid,
                                                        shuffle=False)
    return batch_iterator


def test_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator for validation """
    from msmd.data_pools.batch_iterators import VariableLengthSequencePoolIterator
    batch_iterator = VariableLengthSequencePoolIterator(batch_size=batch_size,
                                                        prepare=prepare_test,
                                                        shuffle=False)
    return batch_iterator


def runtime_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator for runtime: discards the outputs """
    from msmd.data_pools.batch_iterators import VariableLengthSequencePoolIterator
    # Change k_samples to a fixed number to log every k_samples batches. Effectively logs more often.
    batch_iterator = VariableLengthSequencePoolIterator(batch_size=batch_size,
                                                        prepare=prepare_runtime,
                                                        shuffle=False)
    return batch_iterator


##############################################################################


if __name__ == '__main__':
    logging.info('Running model onset_counter in test mode.')

    # Init model
    # ----------

    _batch_size = 3
    max_length = 256
    n_rows = 128
    hidden_size = 16
    n_rnn_layers = 1

    model = BaseConvnet(n_item_rows=n_rows,
                        n_input_channels=1,
                        n_hidden_rnn=16,
                        n_classes_out=MAX_N_SIMULTANEOUS_ONSETS,
                        leaky_relu=False)

    # Prepare dummy batch
    # -------------------


    # container batch
    midi_in = np.zeros((_batch_size, n_rows, max_length))

    # Generate random MIDI input data
    from e2eomr.utils import generate_random_mm
    for i in range(_batch_size):
        mm = generate_random_mm((n_rows, max_length))
        midi_in[i, :, :] = mm

    print('midi_in shape: {}'.format(midi_in.shape))

    X, y = prepare_patch_and_target(None, None, midi_in, None)

    X_torch = Variable(torch.from_numpy(X).float())
    y_torch = Variable(torch.from_numpy(y).float())

    print('X.shape = {}'.format(X.shape))
    print('y.shape = {}'.format(y.shape))

    y_pred = model(X_torch)

    print('y_true = {}'.format(y_torch))
    print('y_pred = {}'.format(y_pred))

