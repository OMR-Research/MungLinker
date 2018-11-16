#!/usr/bin/env python
"""This is a file that implements various evaluation functionality
for the MungLinker experiments."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import time

import numpy as np

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def evaluate_clf(pred_classes, true_classes):
    """Returns binary classification metrics: accuracy overall,
    and precisions, recalls and f-scores as pairs of scores for the 0 and 1
    classes. Typically, the f-score for the positive class is what MuNGLinker
    is most interested in."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    accuracy = accuracy_score(true_classes, pred_classes)
    precision, recall, f_score, true_sum = precision_recall_fscore_support(true_classes,
                                                                           pred_classes)
    return {'acc': accuracy,
            'prec': precision,
            'rec': recall,
            'fsc': f_score,
            'support': true_sum}


def eval_clf_by_class_pair(mungos_from, mungos_to, true_classes, pred_classes,
                           flatten_results=False, retain_negative=False):
    """Produce a dict of evaluation results for individual class pairs
    in the data. (Note that grammar restrictions are already built into
    that, if a grammar is used.) By default, retains only the recall, precision,
    f-score for the positive class, and support for both.

    Expects true_classes and pred_classes to be 1-D numpy arrays.

    :param flatten_results: If set, will flatten the results, so that
        the output is a single-level dict with keys like ``notehead-full__stem__fsc``,
        ``key_signature__sharp__fsc``, etc.

    :param retain_negative: If set, will not discard the negative class result
        in per-class data. [NOT IMPLEMENTED]
    """
    class_pair_index = {}
    for i, (m_fr, m_to, tc, pc) in enumerate(zip(mungos_from, mungos_to,
                                                 true_classes, pred_classes)):
        cpair = m_fr.clsname, m_to.clsname
        if cpair not in class_pair_index:
            class_pair_index[cpair] = []
        class_pair_index[cpair].extend(i)

    class_pair_results = dict()
    for cpair in class_pair_index:
        cpi = np.array(class_pair_index[cpair])
        cp_true = true_classes[cpi]
        cp_pred = pred_classes[cpi]
        cp_results_all = evaluate_clf(cp_pred, cp_true)
        cp_results = {
            'rec': cp_results_all['rec'][1],
            'prec': cp_results_all['prec'][1],
            'fsc': cp_results_all['fsc'][1],
            'support': cp_results_all['support'],
            'loss': cp_results_all['loss']}
        if flatten_results:
            cpair_name = '__'.join(*cpair)
            for k, v in cp_results.items():
                class_pair_results[cpair_name + '__' + k] = v
        else:
            class_pair_results[cpair] = cp_results

    return class_pair_results


def print_class_pair_results(class_pair_results, min_support=100):
    """Prints the class pair results ordered by support, from more to less.
    Prints only class pairs that have at least ``min_support`` positive
    plus negative examples."""
    cpair_ordered = sorted(class_pair_results.keys(),
                           key=lambda cp: sum(class_pair_results[cp]['support']),
                           reversed=True)
    for cpair in cpair_ordered:
        values = class_pair_results[cpair]
        if sum(values['support']) < min_support:
            continue
        cpair_name = '__'.join(*cpair)
        for k in ['rec', 'prec', 'fsc', 'support', 'loss']:
            print('{}__{}'.format(cpair_name, k), values[k])



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
