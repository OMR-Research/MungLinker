#!/usr/bin/env python
"""This is a script that..."""
from __future__ import print_function, unicode_literals
import logging

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


MIDI_MAX_LEN = 256
MIDI_N_PITCHES = 128

BATCH_SIZE = 4


class MultitaskClassFeedback(nn.Module):
    raise NotImplementedError()


def get_build_model():
    return MultitaskClassFeedback


##############################################################################


def prepare_patch_and_target(mungos_from, mungos_to, patches, targets,
                             target_is_onehot=True):
    """Does not do anything to patches.

    For the multitask learning, we intend to use the class names of mungos_from
    and mungos_to (which is why the data pool is giving these to us).

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

    model = MultitaskFullyShared(n_input_channels=patch_channels,
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

