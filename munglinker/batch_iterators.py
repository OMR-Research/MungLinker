import copy

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

    def __init__(self, batch_size, transform=None, k_samples=None, shuffle=True):
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle

        self.k_samples = k_samples
        self.epoch_counter = 0
        self.n_epochs = None

    def __call__(self, pool):
        self.pool = pool
        if self.k_samples is None:
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
            # if len(pool_items["mungos_from"]) < self.batch_size:
            #     n_missing = self.batch_size - len(pool_items[0])
            #
            #     additional_pool_items = self.pool[0:n_missing]
            #
            #     # Batch building: this is one spot that used to be
            #     # pool-specific, but now the BatchIterator can deal with
            #     # combining arbitrary pool outputs as long as they consist
            #     # of a combination of lists and numpy arrays.
            #     pool_items = self.combine_pool_items(pool_items, additional_pool_items)

            if self.transform is None:
                yield pool_items
            else:
                yield self.transform(*pool_items)

        self.epoch_counter += 1


    def combine_pool_items(self, *pool_items_list):
        """Pool items are a tuple of entity batches. Supported "entity"
        types are a list, or a numpy array."""
        if len(pool_items_list) == 0:
            return None
        if len(pool_items_list) == 1:
            return pool_items_list[0]

        n_entities = len(pool_items_list[0])

        for pool_items in pool_items_list:
            if len(pool_items) != n_entities:
                raise ValueError('Batch iterator cannnot combine pool items'
                                 ' with different entity counts: {}'
                                 ''.format(pool_items_list))

        output = [copy.deepcopy(item) for item in pool_items_list[0]]
        for item_idx, pool_item in enumerate(pool_items_list[1:]):
            for e_idx, entity in enumerate(pool_item):
                if isinstance(output[e_idx], list):
                    if not isinstance(entity, list):
                        raise TypeError('Entities have inconsistent types:'
                                        ' in item 0 entity {} is a list, in item'
                                        ' {} it is {}'.format(e_idx, item_idx + 1,
                                                              type(entity)))
                    output[e_idx].extend(entity)
                elif isinstance(output[e_idx], np.ndarray):
                    if not isinstance(entity, np.ndarray):
                        raise TypeError('Entities have inconsistent types:'
                                        ' in item 0 entity {} is a ndarray, in item'
                                        ' {} it is {}'.format(e_idx, item_idx + 1,
                                                              type(entity)))
                    output[e_idx] = np.concatenate((output[e_idx], entity))

        return output
