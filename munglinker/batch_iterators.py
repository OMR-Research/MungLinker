#!/usr/bin/env python
"""This is a script that..."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import time

import numpy as np

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################

# Code from Matthias Dorfer.

# This takes care of multi-threading data loading.
def threaded_generator(generator, num_cached=10):
    """
    Threaded generator
    """
    try:
        import Queue
    except ImportError:
        import queue as Queue
    queue = Queue.Queue(maxsize=num_cached)
    queue = Queue.Queue(maxsize=num_cached)
    end_marker = object()

    # define producer
    def producer():
        for item in generator:
            #item = np.array(item)  # if needed, create a copy here
            queue.put(item)
        queue.put(end_marker)

    # start producer
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()


def generator_from_iterator(iterator):
    """
    Compile generator from iterator
    """
    for x in iterator:
        yield x


def threaded_generator_from_iterator(iterator, num_cached=10):
    """
    Compile threaded generator from iterator
    """
    generator = generator_from_iterator(iterator)
    return threaded_generator(generator, num_cached)


# --- class definitions ---


class PoolIterator(object):
    """
    Batch iterators are used from the by the model as the interface between
    a data pool that provides the training entities and what the model requires.
    The data pool is supposed to provide data points; the model's prepare()
    functions define how these data points are to be shown to the model during
    training/runtime, and the batch iterator is an efficient interface that
    feeds data points to the models' prepare() smoothly (esp. for training).
    """
    def __init__(self, batch_size, prepare=None, k_samples=None, shuffle=True):
        self.batch_size = batch_size

        if prepare is None:
            def prepare(x, y):
                return x, y
        self.prepare = prepare
        self.shuffle = shuffle

        self.k_samples = k_samples
        self.epoch_counter = 0
        self.n_epochs = None

    def __call__(self, pool):
        self.pool = pool
        if self.k_samples is None:
            self.k_samples = self.pool.shape[0]
        self.n_batches = self.k_samples // self.batch_size
        self.n_epochs = max(1, self.pool.shape[0] // self.k_samples)

        return self

    def __iter__(self):
        n_samples = self.k_samples
        bs = self.batch_size

        # compute current epoch index
        idx_epoch = np.mod(self.epoch_counter, self.n_epochs)

        for i in range((n_samples + bs - 1) / bs):

            i_start = i * bs + idx_epoch * self.k_samples
            i_stop = (i + 1) * bs + idx_epoch * self.k_samples
            sl = slice(i_start, i_stop)
            xb, yb = self.pool[sl]

            # When the data pool runs out: re-draw from beginning
            if xb.shape[0] < self.batch_size:
                n_missing = self.batch_size - xb.shape[0]

                x_con, y_con = self.pool[0:n_missing]

                # Batch building: this is one spot that might need
                # to get overriden if what the data pool provieds
                # are not numpy-concatenable this way.
                xb = np.concatenate((xb, x_con))
                yb = np.concatenate((yb, y_con))

            yield self.transform(xb, yb)

        self.epoch_counter += 1

        # shuffle train data after full set iteration
        if self.shuffle and (idx_epoch + 1) == self.n_epochs:
            self.pool.reset_batch_generator()

    def transform(self, *args, **kwargs):
        return self.prepare(*args, **kwargs)


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    # Your code goes here
    raise NotImplementedError()

    _end_time = time.clock()
    logging.info('[XXXX] done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
