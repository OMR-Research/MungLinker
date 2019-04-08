from torch import nn as nn

from munglinker.batch_iterators import PoolIterator


class MungLinkerNetwork(nn.Module):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size


    def train_batch_iterator(self):
        return PoolIterator(batch_size=self.batch_size, transform=None, shuffle=True)

    def valid_batch_iterator(self):
        return PoolIterator(batch_size=self.batch_size, transform=None, shuffle=False)

    def test_batch_iterator(self):
        return PoolIterator(batch_size=self.batch_size, transform=None, shuffle=False)

    def runtime_batch_iterator(self):
        """ Compile batch iterator for runtime: discards the outputs """
        # Change k_samples to a fixed number to log every k_samples batches. Effectively logs more often.
        return PoolIterator(batch_size=self.batch_size, transform=None, shuffle=False)
