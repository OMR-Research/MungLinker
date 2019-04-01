from abc import abstractmethod
from typing import List

import numpy as np
from muscima.cropobject import CropObject
from torch import nn as nn

from munglinker.batch_iterators import PoolIterator


class MungLinkerNetwork(nn.Module):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    @abstractmethod
    def prepare_train(self, *args, **kwargs):
        pass

    @abstractmethod
    def prepare_valid(self, *args, **kwargs):
        pass

    @abstractmethod
    def prepare_test(self, *args, **kwargs):
        pass

    @abstractmethod
    def prepare_runtime(self, *args, **kwargs):
        """At runtime, we return the MuNGOs explicitly, so that it is
        straightforward to pair the prediction result without having
        to deal with matching the predictions back to the data pool's
        training entities.
        """
        pass

    def train_batch_iterator(self):
        return PoolIterator(batch_size=self.batch_size, prepare=self.prepare_train, shuffle=True)

    def valid_batch_iterator(self):
        return PoolIterator(batch_size=self.batch_size, prepare=self.prepare_valid, shuffle=False)

    def test_batch_iterator(self):
        return PoolIterator(batch_size=self.batch_size, prepare=self.prepare_test, shuffle=False)

    def runtime_batch_iterator(self):
        """ Compile batch iterator for runtime: discards the outputs """
        # Change k_samples to a fixed number to log every k_samples batches. Effectively logs more often.
        return PoolIterator(batch_size=self.batch_size, prepare=self.prepare_runtime, shuffle=False)
