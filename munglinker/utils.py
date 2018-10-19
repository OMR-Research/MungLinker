"""This module implements a class that..."""
from __future__ import print_function, unicode_literals

import copy
import logging
import os

import pickle

import collections

import datetime
import numpy as np
from muscima.grammar import DependencyGrammar
from muscima.io import parse_cropobject_class_list

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


# init color printer
class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        """ Constructor """
        pass

    def print_colored(self, string, color):
        """ Change color of string """
        return color + string + BColors.ENDC


PRIMITIVES = ["16th_rest",
              "32th_rest",
              "64th_and_higher_rest",
              "8th_rest",
              "barlines",
              "beam",
              "c-clef",
              "double_flat",
              "double_sharp",
              "duration-dot",
              "f-clef",
              "flags",
              "flat",
              "g-clef",
              "half_rest",
              "ledger_line",
              "natural",
              "noteheads",
              "quarter_rest",
              "sharp",
              "stem",
              "whole_rest"]

NOTE_PRIMITIVES = ["noteheads", "stem", "beam", "flags"]

CLEF_PRIMITIVES = ["c-clef", "g-clef", "f-clef"]

RESULTS_META_ROOT = '/Users/hajicj/jku/omr_baseline_matthias'
MULTICHANNEL_ROOT = '/Users/hajicj/jku/omr_baseline_matthias'


##############################################################################


def build_experiment_name(args):

    split = os.path.splitext(os.path.basename(args.split_file))[0]
    config = os.path.splitext(os.path.basename(args.config_file))[0]
    model = args.model

    # if hasattr(args, 'exclude_images') and args.exclude_images:
    #     testset = os.path.split(os.path.splitext(args.exclude_images)[0])[-1].split('-')[1]
    # elif hasattr(args, 'testset'):
    #     testset = args.testset
    # else:
    #     testset = 'unspecified'

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = 'E2EOMR_{0}_{1}_{2}' \
           ''.format(split,
                     config,
                     # testset,
                     timestamp)

    # if args.zoom != 0:
    #     name += '_z{0:.2f}'.format(args.zoom)
    # if args.multiencoder_model:
    #     name += '_multiEnc'
    if args.augmentation:
        name += '_aug'
    # if hasattr(args, 'model'):
    #     model_name = os.path.splitext(os.path.basename(args.model))[0]
    #     name += '_{0}'.format(model_name)
    if args.exp_tag:
        name += '_{0}'.format(args.exp_tag)
    return name


class ResultsServer(object):
    """Filesystem abstraction over results dir"""
    valid_testsets = ['indep', 'dep']
    valid_thresholds = [0.25, 0.5, 0.75]
    valid_detectors = ['mask', 'conv_hull', 'combined']

    detector_base_colors = {'mask': '#1111aa',
                            'conv_hull': '#11aa11',
                            'combined': '#00aaaa'}

    def __init__(self, root):
        self.root = root

    def load_results(self, primitive, testset, threshold, detector):
        """
        :param primitive: One of the primitives for which we have results.
        :param testset:  One of "dep", "indep"
        :param threshold: One of "0.25", "0.50", "0.75"
        :param detector: One of "mask", "conv_hull", "combined"
        :return: Dict with results.
        """
        thr = '{0:.2f}'.format(threshold)
        fname = 'results_{3}_labels_{2}_{1}_{0}.pkl'.format(primitive, testset, thr, detector)
        filename = os.path.join(self.root, fname)

        if not os.path.isfile(filename):
            raise ValueError('Requested non-existent results file: {0}'.format(filename))

        with open(filename, 'rb') as hdl:
            res = pickle.load(hdl, encoding='latin1')

        return res

    def find_best(self, primitive, testsets=None, thresholds=None, detectors=None):
        """Finds the best configuration for the primitives, measured by f-score.
        Returns the given combination of ``(testset, threshold, output)``, so that
        it can be passed into ``load_results()``.

        You can specify constraints through testsets, thresholds, and outputs
        (e.g., "Find the best configuration for this primitive when the threshold
        is set to 0.5 or 0.75").
        """
        best = -1.0
        best_combination = None

        if testsets is None:
            testsets = self.valid_testsets
        if thresholds is None:
            thresholds = self.valid_thresholds
        if detectors is None:
            detectors = self.valid_detectors

        for testset in testsets:
            for thr in thresholds:
                for output in detectors:
                    res = self.load_results(primitive, testset, thr, output)
                    f = res['f1']
                    if f > best:
                        best = f
                        best_combination = testset, thr, output
        return best_combination


def collect_and_plot_results(results_root, primitives=PRIMITIVES,
                 testset='indep',
                 thresholds=(0.25, 0.5, 0.75),
                 detectors=('mask', 'combined', 'conv_hull')):
    """Show the results for the given primitives on the given test set.
    The thresholds and outputs can optionally be supplied to restrict the results
    to a subset of the possible values.

    :param results_root: Where to read the result dicts

    :param primitives: The list of primitives for which to plot the results.
        By default, just takes the

    :param testset: One of "dep", "indep". By default, it is "indep".
    """
    all_results = collect_results(results_root, primitives, detectors, testset, thresholds)
    plot_results(all_results, detectors, testset)


def collect_results(results_root,
                    primitives=PRIMITIVES,
                    detectors=('mask', 'combined', 'conv_hull'),
                    testset='indep',
                    thresholds=(0.25, 0.5, 0.75)):
    # Collect results
    r_server = ResultsServer(root=results_root)
    all_results = dict()
    all_thrs = dict()
    for p in primitives:
        p_results = dict()
        p_thrs = dict()
        for d in detectors:
            best_thr = 0.25
            best_thr_f1 = 0.0
            for thr in thresholds:
                try:
                    res = r_server.load_results(p, testset, thr, d)
                except ValueError:
                    continue
                if res['f1'] > best_thr_f1:
                    best_thr_f1 = res['f1']
                    best_thr = thr
            p_results[d] = best_thr_f1
            p_thrs[d] = best_thr
        # Only include primitives with meaningful results.
        if max(p_results.values()) > 0:
            all_results[p] = copy.deepcopy(p_results)
            all_thrs[p] = copy.deepcopy(p_thrs)
    return all_results


def plot_results(all_results, detectors, testset):
    # Plot them
    import matplotlib.pyplot as plt
    detector_colors = [ResultsServer.detector_base_colors[d] for d in detectors]
    min_val = 1.0
    max_val = 0.0
    xticklabels = []
    xticks = []
    for i, p in enumerate(sorted(all_results.keys())):
        p_results = [all_results[p][d] for d in detectors]

        if min(p_results) == max(p_results) == 0:
            continue

        x_vals = np.asarray(range(len(detectors))) * 0.8 + (len(detectors) + 1) * i
        xticks.append(x_vals[len(x_vals) // 2])
        xticklabels.append(p)
        plt.bar(x_vals, p_results, color=detector_colors)

        min_val = min(min_val, min(p_results))
        max_val = max(max_val, max(p_results))
    plt.ylim(min_val - 0.1, 1.05)
    plt.xticks(xticks, xticklabels, rotation='vertical')
    if testset == 'indep':
        testset_name = "Writer-Independent"
    elif testset == 'dep':
        testset_name = "Writer-Dependent"
    plt.title('F1 score: {0}'.format(testset_name))
    plt.tight_layout()
    plt.show()


def plot_results_difference(results_base, results_other, testset, name='NONAME'):
    """After collect_results with

    :param results_base:
    :param results_other:
    :param testset:
    :return:
    """
    import matplotlib.pyplot as plt
    min_val = 1.0
    max_val = -1.0
    xticklabels = []
    xticks = []
    for i, p in enumerate(sorted(results_other.keys())):
        b_result = max(results_base[p].values())
        o_result = max(results_other[p].values())
        delta_result = o_result - b_result

        x_val = 1.05 * i
        # x_vals = np.asarray(range(len(detectors))) * 0.8 + (len(detectors) + 1) * i
        xticks.append(x_val)
        xticklabels.append(p)
        if delta_result > 0:
            color = '#00aa00'
        else:
            color = '#ff3333'
        plt.bar([x_val], [delta_result], color=color)
        min_val = min(min_val, delta_result)
        max_val = max(max_val, delta_result)

    print('min val: {0}, max val: {1}'.format(min_val, max_val))
    plt.ylim(min_val - 0.1, max_val + 0.1)
    plt.xticks(xticks, xticklabels, rotation='vertical')
    if testset == 'indep':
        testset_name = "Writer-Independent"
    elif testset == 'dep':
        testset_name = "Writer-Dependent"
    plt.title('F1 diff: {0}, {1}'.format(testset_name, name))
    plt.tight_layout()
    plt.show()


##############################################################################
# Little utilities for patches

def generate_random_patches_batch(batch_size, patch_shape):
    patch_in = np.zeros((batch_size, *patch_shape))
    for i in range(batch_size):
        p = generate_random_patch(patch_shape)
        patch_in[i] = p
    return patch_in


def generate_random_patch(shape, is_binary=True):
    output = np.random.randint(0, 2, shape, dtype='uint8')
    import cv2
    output[0] = cv2.morphologyEx(output[0], op=cv2.MORPH_DILATE, kernel=np.ones((2, 3)))
    output[0] = cv2.morphologyEx(output[0], op=cv2.MORPH_OPEN, kernel=np.ones((21, 21)))
    output[1][output[0] == 0.0] = 0.0
    output[2][output[0] == 0.0] = 0.0
    output[output > 0] = 1.0
    return output.astype('float32')


def get_dummy_target(patch):
    if patch[0].sum() < (patch[0].size / 6.0):
        return 1
    else:
        return 0


def generate_munglinker_training_batch(batch_size, patch_shape):
    patches = generate_random_patches_batch(batch_size, patch_shape)
    targets = np.array([get_dummy_target(p) for p in patches])
    return patches, targets


##############################################################################
# Target-side utilities

def targets2classes(targets):
    """From the two-class softmax outputs, creates a binary vector: 0 for no
    edge, 1 for edge.

    >>> import numpy as np
    >>> x = np.array([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6], [0.5, 0.5]])
    >>> targets2classes(x)
    array([0, 1, 1, 0])
    >>> y = np.array([1, 0, 0, 1, 1])
    >>> targets2classes(y)
    array([1, 0, 0, 1, 1])

    """
    if targets.ndim == 1:
        return targets
    output = np.argmax(targets, axis=1)
    return output


##############################################################################
# Little utilities for MIDI matrix

def midi_matrix_to_midi(midi_matrix, FPS=20, tempo=120):
    """Returns a midiutils.MidiFile.MIDIFile object."""
    # Use MIDIBuilder.build_midi()
    from muscima.inference import MIDIBuilder
    builder = MIDIBuilder()
    pitches, durations, onsets = builder.midi_matrix_to_pdo(midi_matrix,
                                                            framerate=FPS,
                                                            tempo=tempo)
    midi = builder.build_midi(pitches, durations, onsets,
                              selection=None, tempo=120)
    return midi


def generate_random_mm(shape, max_duration_frames=40, onset_density=0.001):
    """Generates a random MIDI matrix parametrized by onset density and maximum
    duration. Each cell will be an onset cell with ``onset_density`` chance,
    and its duration will be uniformly drawn from (1, max_duration_frames)."""
    onsets_matrix = (np.random.rand(*shape) <= onset_density).astype('uint8')
    midi_matrix = np.zeros(shape, dtype='uint8')

    n_onsets = onsets_matrix.sum()

    for x, y in zip(*onsets_matrix.nonzero()):
        duration = np.random.randint(0, max_duration_frames)
        midi_matrix[x, y:y+duration+1] = 1

    print('Built random MIDI matrix of shape {}: total onsets {}, total nonzero entries {} / {}'
          ''.format(shape, n_onsets, midi_matrix.sum(), midi_matrix.size))
    return midi_matrix


def n_onsets_from_midi_matrix(mm):
    """Counts MIDI onset cells.

    >>> m = np.array([[1, 0, 0, 1, 1, 1, 1, 0],
    ...               [0, 0, 1, 0, 0, 0, 0, 1],
    ...               [1, 1, 1, 1, 1, 1, 1, 1],
    ...               [0, 0, 0, 0, 1, 1, 1, 1],
    ...               [1, 1, 1, 1, 0, 0, 0, 0]])
    >>> n_onsets_from_midi_matrix(m)
    7
    """
    n_onsets = 0
    n_onsets += mm[:, 0].sum()
    n_onsets += ((mm[:, 1:] - mm[:, :-1]) == 1).astype(np.int).sum()
    return n_onsets


##############################################################################


class MockNetwork(object):
    """This class is a mock object for testing pipelines when no real trained
    model is available. Outputs random labels.
    """
    def predict(self, data_pool, runtime_batch_iterator):
        # Ensure correct data pool behavior.
        data_pool.resample_train_entities = False

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
        all_np_pred_classes = targets2classes(all_np_preds)
        return all_mungo_pairs, all_np_pred_classes


##############################################################################


def select_model(model_path):
    """Select model (returns the module). If the ``MOCK`` model is selected,
    returns not a model module, but the MockNetwork object that simulates
    giving predictions like a ``PyTorchNetwork.predict()`` would."""
    model_str = os.path.basename(model_path)
    model_str = model_str.split('.py')[0]
    import importlib
    model = importlib.import_module('munglinker.models.{}'.format(model_str))
    # exec('from munglinker.models import ' + model_str + ' as model')
    model.EXP_NAME = model_str
    return model


def cuda_works():
    import torch
    if not torch.cuda.is_available():
        return False

    x = np.ones((10, 10))
    from torch.autograd import Variable
    x_torch = Variable(torch.from_numpy(x))
    try:
        x_torch.cuda()
    except RuntimeError:
        return False

    return True

##############################################################################
# Visualizations


def plot_batch_patches(X, y, max_items=6):
    """Shows the current input batch patches; true/false is labeled
    on the Y axis of each subplot."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import center_of_mass

    plt.figure(figsize=(9.0, 6.0))
    plt.clf()

    n_items = min(X.shape[0], max_items)
    n_items_per_row = 3
    n_rows = int(np.ceil(n_items / n_items_per_row))

    for i in range(min(X.shape[0], max_items)):
        plt.subplot(n_rows, n_items_per_row, i+1)
        patch = X[i]
        target = y[i]

        patch_sum = np.sum(patch, axis=0)
        plt.imshow(patch_sum, cmap='gray', interpolation='nearest')

        # Indicate from & to
        cfx, cfy = center_of_mass(patch[1])
        ctx, cty = center_of_mass(patch[2])
        arrowcolor = 'r'
        if y[i][1] > y[i][0]:
            arrowcolor = 'g'
        plt.arrow(cfy, cfx, cty - cfy, ctx - cfx,
                  color=arrowcolor,
                  width=0.1, head_width=15, head_length=20,
                  length_includes_head=True, overhang=0.5)

        plt.ylabel(target)
        plt.yticks([])
        plt.xticks([])

    plt.tight_layout()
    plt.show()


def show_batch_simple(X, y, max_items=3):
    """Shows the current input batch with its outputs."""
    n_rows = min(max_items, X.shape[0])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.clf()

    for i, (x_i, y_i) in enumerate(zip(X, y)):
        if i >= max_items:
            break
        plt.subplot(n_rows, 1, i+1)
        plt.imshow(x_i[0], cmap='gray', origin='upper', aspect='auto', interpolation='nearest')
        plt.xlabel(i)
        plt.ylabel(y_i)

    plt.show()


def show_onset_counter_predictions(X_var, y_true_var, net, max_items=1):

    # Get all network intermediate steps
    y_pred = net(X_var)
    conv_out = net.get_conv_output(X_var)
    batch_size, n_channels, n_rows, n_frames = conv_out.size()
    softmax_outs = net.conv_out2softmax_out(conv_out)
    framewise_counts = net.softmax2framewise_onset_sums(softmax_outs, n_frames=n_frames)
    total_counts = framewise_counts.sum(dim=1)

    # n_show = min(max_items, X.shape[0])
    # ...just plot one instance, for now

    import matplotlib.pyplot as plt
    plt.figure()
    plt.clf()

    softmax_outs_np = softmax_outs.data.numpy()
    framewise_counts_np = framewise_counts.data.numpy()
    total_counts_np = total_counts.data.numpy()

    print('Softmax out shape = {}'.format(softmax_outs_np.shape))
    print('Framewise count shape = {}'.format(framewise_counts_np.shape))

    softmax_out = softmax_outs_np[0]
    framewise_count = framewise_counts_np[0]
    total_count = total_counts_np[0]

    X_np = X_var.data.numpy()

    plt.subplot(2, 1, 1)
    plt.imshow(X_np[0][0], cmap='gray', origin='upper', aspect='auto', interpolation='nearest')
    plt.xlabel('Input image')
    plt.ylabel('True: {}'.format(y_true_var[0]))

    plt.subplot(2, 1, 2)
    plt.imshow(softmax_out[:, :, 0], origin='upper', aspect='auto', interpolation='nearest')
    plt.xlabel('Softmax outputs per RNN frame')
    plt.yticks(list(range(net.n_classes_out)))
    #
    # plt.subplot(3, 1, 3)
    # plt.imshow(framewise_count, origin='upper', aspect='auto', interpolation='nearest')
    # plt.xlabel('Onset counts: predicted {}, true {} '.format(total_count, y_true_var[0]))
    # plt.ylabel('')

    plt.show()


def show_onset_sequence_predictions(X_var, y_true_var, net, max_items=1):

    # Get all network intermediate steps
    y_pred = net(X_var)
    conv_out = net.get_conv_output(X_var)
    batch_size, n_channels, n_rows, n_frames = conv_out.size()
    softmax_outs = net.conv_out2softmax_out(conv_out)
    framewise_counts = net.softmax2framewise_onset_sums(softmax_outs, n_frames=n_frames)
    total_counts = framewise_counts.sum(dim=1)

    # print('y_ture_var shape: {}'.format(y_true_var.size()))

    # n_show = min(max_items, X.shape[0])
    # ...just plot one instance, for now

    import matplotlib.pyplot as plt
    plt.figure()
    plt.clf()

    softmax_outs_np = softmax_outs.data.numpy()
    framewise_counts_np = framewise_counts.data.numpy()
    total_counts_np = total_counts.data.numpy()
    true_framewise_softmax_np = y_true_var.data.numpy()

    # print('Softmax out shape = {}'.format(softmax_outs_np.shape))
    # print('Framewise count shape = {}'.format(framewise_counts_np.shape))

    softmax_out = softmax_outs_np[0]
    softmax_true = true_framewise_softmax_np[0]
    framewise_count = framewise_counts_np[0]
    total_count = total_counts_np[0]

    X_np = X_var.data.numpy()

    plt.subplot(3, 1, 1)
    plt.imshow(X_np[0][0], cmap='gray', origin='upper', aspect='auto', interpolation='nearest')
    plt.xlabel('Input image')
    plt.ylabel('True: {}'.format(y_true_var[0]))

    plt.subplot(3, 1, 2)
    plt.imshow(softmax_true, origin='upper', aspect='auto', interpolation='nearest')
    plt.xlabel('True count table per RNN frame')
    plt.yticks(list(range(net.n_classes_out)))

    plt.subplot(3, 1, 3)
    plt.imshow(softmax_out[:, :, 0], origin='upper', aspect='auto', interpolation='nearest')
    plt.xlabel('Softmax outputs per RNN frame')
    plt.yticks(list(range(net.n_classes_out)))


    #
    # plt.subplot(3, 1, 3)
    # plt.imshow(framewise_count, origin='upper', aspect='auto', interpolation='nearest')
    # plt.xlabel('Onset counts: predicted {}, true {} '.format(total_count, y_true_var[0]))
    # plt.ylabel('')

    plt.show()


def load_grammar(filename):
    mungo_classes_file = os.path.splitext(filename)[0] + '.xml'
    mlclass_dict = {m.name: m for m in parse_cropobject_class_list(mungo_classes_file)}
    g = DependencyGrammar(grammar_filename=filename, alphabet=list(mlclass_dict.keys()))
    return g


def config2data_pool_dict(config):
    """Prepare data pool kwargs from an exp_config dict.

    Grammar file is loaded relative to the munglinker/ package."""
    data_pool_dict = {
        'max_edge_length': config['THRESHOLD_NEGATIVE_DISTANCE'],
        'max_negative_samples': config['MAX_NEGATIVE_EXAMPLES_PER_OBJECT'],
        'resample_train_entities': config['RESAMPLE_EACH_EPOCH'],
        'patch_size': (config['PATCH_HEIGHT'], config['PATCH_WIDTH']),
        'zoom': config['IMAGE_ZOOM'],
        'max_patch_displacement': config['MAX_PATCH_DISPLACEMENT'],
        'balance_samples': False
    }

    if 'PATCH_NO_IMAGE' in config:
        data_pool_dict['PATCH_NO_IMAGE'] = config['PATCH_NO_IMAGE']

    # Load grammar, if requested
    if 'RESTRICT_TO_GRAMMAR' in config:
        if not os.path.isfile(config['RESTRICT_TO_GRAMMAR']):
            grammar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        config['RESTRICT_TO_GRAMMAR'])
        else:
            grammar_path = config['RESTRICT_TO_GRAMMAR']

        if os.path.isfile(grammar_path):
            grammar = load_grammar(grammar_path)
            data_pool_dict['grammar'] = grammar
        else:
            logging.warning('Config contains grammar {}, but it is unreachable'
                            ' both as an absolute path and relative to'
                            ' the munglinker/ package. No grammar loaded.'
                            ''.format(config['RESTRICT_TO_GRAMMAR']))

    return data_pool_dict
