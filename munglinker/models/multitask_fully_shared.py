import logging
from typing import List

import numpy as np
import torch
from muscima.cropobject import CropObject
from torch.autograd import Variable

from munglinker.models.munglinker_network import MungLinkerNetwork


class MultitaskFullyShared(MungLinkerNetwork):
    def prepare_patch_and_target(self, mungos_from: List[CropObject], mungos_to: List[CropObject], patches: np.ndarray,
                                 targets: np.ndarray, target_is_onehot: bool = False):
        raise NotImplementedError()

    def prepare_train(self, *args, **kwargs):
        raise NotImplementedError()

    def prepare_valid(self, *args, **kwargs):
        raise NotImplementedError()

    def prepare_test(self, *args, **kwargs):
        raise NotImplementedError()

    def prepare_runtime(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *input):
        raise NotImplementedError()


if __name__ == '__main__':
    logging.info('Running model onset_counter in test mode.')

    _batch_size = 6
    patch_shape = 3, 256, 512
    patch_height = 256
    patch_width = 512
    patch_channels = 3

    from munglinker.utils import generate_munglinker_training_batch, plot_batch_patches

    model = MultitaskFullyShared(4)
    patches, targets = generate_munglinker_training_batch(_batch_size, patch_shape)

    print('patches shape: {}'.format(patches.shape))

    X, y = model.prepare_patch_and_target([], [], patches, targets)

    plot_batch_patches(patches, targets)

    X_torch = Variable(torch.from_numpy(X).float())
    y_torch = Variable(torch.from_numpy(y).float())

    print('X.shape = {}'.format(X.shape))
    print('y.shape = {}'.format(y.shape))

    y_pred = model(X_torch)

    print('y_true = {}'.format(y_torch))
    print('y_pred = {}'.format(y_pred))
