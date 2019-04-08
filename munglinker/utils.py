import datetime
import logging
import os

import numpy
import numpy as np
from muscima.grammar import DependencyGrammar
from muscima.io import parse_cropobject_class_list

from munglinker.models.base_convnet import BaseConvnet
from munglinker.models.base_convnet_double_filters import BaseConvnetDoubleFilters
from munglinker.models.mock_convnet import MockNetwork
from munglinker.models.multitask_class_feedback import MultitaskClassFeedback
from munglinker.models.multitask_fully_shared import MultitaskFullyShared
from munglinker.models.multitask_partially_shared import MultitaskPartiallyShared
from munglinker.models.munglinker_network import MungLinkerNetwork


def build_experiment_name(args):
    split = os.path.splitext(os.path.basename(args.split_file))[0]
    config = os.path.splitext(os.path.basename(args.config_file))[0]
    model = args.model

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = 'E2EOMR_{0}_{1}_{2}' \
           ''.format(split,
                     config,
                     # testset,
                     timestamp)

    if args.augmentation:
        name += '_aug'
    if args.exp_tag:
        name += '_{0}'.format(args.exp_tag)
    return name


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

def targets2classes(targets, threshold=0.5):
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
        return (targets >= threshold).astype('uint8')
    if targets.ndim == 2 and targets.shape[-1] == 1:
        return (targets >= threshold).astype('uint8')
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
        midi_matrix[x, y:y + duration + 1] = 1

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


def select_model(model_name: str, batch_size: int) -> MungLinkerNetwork:
    """Select model (returns the module). If the ``MOCK`` model is selected,
    returns not a model module, but the MockNetwork object that simulates
    giving predictions like a ``PyTorchNetwork.predict()`` would."""
    if model_name == "base_convnet":
        return BaseConvnet(batch_size=batch_size)
    elif model_name == "base_convnet_double_filters":
        return BaseConvnetDoubleFilters(batch_size=batch_size)
    elif model_name == "multitask_class_feedback":
        return MultitaskClassFeedback(batch_size=batch_size)
    elif model_name == "multitask_fully_shared":
        return MultitaskFullyShared(batch_size=batch_size)
    elif model_name == "multitask_partially_shared":
        return MultitaskPartiallyShared(batch_size=batch_size)
    elif model_name == "mock":
        return MockNetwork(batch_size=batch_size)
    else:
        raise Exception("Unknown network model selected")


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
        plt.subplot(n_rows, n_items_per_row, i + 1)
        patch = X[i]
        target = y[i]

        patch_sum = np.sum(patch, axis=0)
        plt.imshow(patch_sum, cmap='gray', interpolation='nearest')

        # Indicate from & to
        cfx, cfy = center_of_mass(patch[1])
        ctx, cty = center_of_mass(patch[2])
        arrowcolor = 'r'
        if y.shape[-1] == 2:
            if y[i][1] > y[i][0]:
                arrowcolor = 'g'
            else:
                if y[i] > 0.5:
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
        plt.subplot(n_rows, 1, i + 1)
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
        'patch_size': (config['PATCH_HEIGHT'], config['PATCH_WIDTH']),
        'zoom': config['IMAGE_ZOOM']
    }

    if 'PATCH_NO_IMAGE' in config:
        data_pool_dict['patch_no_image'] = config['PATCH_NO_IMAGE']

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


def dice(Seg, GT):
    """ compute dice coefficient between current segmentation result and groundtruth (GT)"""

    sum_GT = numpy.sum(GT)
    sum_Seg = numpy.sum(Seg)

    if (sum_GT + sum_Seg) == 0:
        dice = 1.0
    else:
        dice = (2.0 * numpy.sum(Seg[GT == 1])) / (sum_Seg + sum_GT)

    return dice
