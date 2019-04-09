import copy
import itertools
from typing import List, Dict

import numpy as np


class PoolIterator(object):
    """
    Batch iterators are used from the by the model as the interface between
    a data pool that provides the training entities and what the model requires.
    The data pool is supposed to provide data points; the model's prepare()
    functions define how these data points are to be shown to the model during
    training/runtime, and the batch iterator is an efficient interface that
    feeds data points to the models' prepare() smoothly (esp. for training).
    """

    def __init__(self, batch_size, transform=None, shuffle=True):
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle

        self.epoch_counter = 0
        self.n_epochs = None

    def __call__(self, pool):
        self.pool = pool
        self.k_samples = len(self.pool)
        self.n_batches = self.k_samples // self.batch_size
        self.n_epochs = max(1, len(self.pool) // self.k_samples)

        return self

    def __iter__(self):
        n_samples = self.k_samples
        bs = self.batch_size

        # compute current epoch index
        idx_epoch = np.mod(self.epoch_counter, self.n_epochs)

        # shuffle train data before each epoch
        if self.shuffle and idx_epoch == 0 and not self.epoch_counter == 0:
            self.pool.shuffle_batches()

        for i in range(int((n_samples + bs - 1) / bs)):
            i_start = i * bs + idx_epoch * self.k_samples
            i_stop = (i + 1) * bs + idx_epoch * self.k_samples
            sl = slice(i_start, i_stop)
            pool_items = self.pool[sl]

            # When the data pool runs out: re-draw from beginning
            if len(pool_items["mungos_from"]) < self.batch_size:
                n_missing = self.batch_size - len(pool_items["mungos_from"])
                additional_pool_items = self.pool[0:n_missing]

                # Batch building: this is one spot that used to be
                # pool-specific, but now the BatchIterator can deal with
                # combining arbitrary pool outputs as long as they consist
                # of a combination of lists and numpy arrays.
                pool_items = self.collate_fn([pool_items, additional_pool_items])

            if self.transform is None:
                yield pool_items
            else:
                yield self.transform(*pool_items)

        self.epoch_counter += 1

    def collate_fn(self, samples: List[Dict]):
        mungos_from = list(itertools.chain.from_iterable([sample["mungos_from"] for sample in samples]))
        mungos_to = list(itertools.chain.from_iterable([sample["mungos_to"] for sample in samples]))
        patches_batch = np.concatenate([sample["patches"] for sample in samples])
        targets = np.concatenate([sample["targets"] for sample in samples])
        return dict(mungos_from=mungos_from, mungos_to=mungos_to, patches=patches_batch, targets=np.array(targets))
