import logging
from typing import List

import numpy as np
from mung.node import Node

from munglinker.models.munglinker_network import MungLinkerNetwork


class MockNetwork(MungLinkerNetwork):
    """This class is a mock object for testing pipelines when no real trained
    model is available. Outputs random labels.
    """

    def prepare_patch_and_target(self, mungos_from: List[Node], mungos_to: List[Node], patches: np.ndarray,
                                 targets: np.ndarray, target_is_onehot: bool = False):
        pass

    def prepare_train(self, *args, **kwargs):
        pass

    def prepare_valid(self, *args, **kwargs):
        pass

    def prepare_test(self, *args, **kwargs):
        pass

    def prepare_runtime(self, *args, **kwargs):
        pass

    def forward(self, *input):
        pass

    def predict(self, data_pool, runtime_batch_iterator):
        # Initialize data feeding from iterator
        iterator = runtime_batch_iterator(data_pool)
        generator = (x for x in iterator)  # No threaded generator

        n_batches = len(data_pool) // runtime_batch_iterator.batch_size
        logging.info('n. of runtime entities: {}; batches: {}'
                     ''.format(len(data_pool), n_batches))

        # Aggregate results
        all_mungo_pairs = []
        all_np_preds = np.array([])

        # Run generator
        for batch_idx, _data_point in enumerate(generator):
            mungo_pairs, np_inputs = _data_point  # next(generator)
            mungo_pairs = list(mungo_pairs)
            all_mungo_pairs.extend(list(mungo_pairs))

            np_pred = np.random.randint(0, 2, len(mungo_pairs))
            # inputs = self._np2torch(np_inputs)
            # pred = self.net(inputs)
            # np_pred = self._torch2np(pred)
            all_np_preds = np.concatenate((all_np_preds, np_pred))

        logging.info('All np preds: {} positive ({})'
                     ''.format(all_np_preds.sum(), all_np_preds.mean()))

        from munglinker.utils import targets2classes

        all_np_pred_classes = targets2classes(all_np_preds)
        return all_mungo_pairs, all_np_pred_classes
