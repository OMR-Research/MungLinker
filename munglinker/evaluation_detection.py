"""This module implements evaluation procedures for symbol detection.
Mostly derived from lasagne_wrapper by Matthias Dorfer: segmentation_utils.py."""
from __future__ import print_function, unicode_literals, division

import argparse
import logging

import collections
import os

import numpy
import time

import pickle
import torch
from scipy.misc import imread
from skimage.measure import label, regionprops

from fcnomr.detector import ConnectedComponentDetector
from fcnomr.model import FCN, apply_model, FCNEncoder, FCNDecoder, MultiEncoderFCN, init_model
from fcnomr.preprocessing import MUSCIMALabelsDataset, MUSCIMALabelIterator
from fcnomr.utils import lasagne_fcn_2_pytorch_fcn, build_experiment_name, label_img2mobcsv

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


def dice(Seg, GT):
    """ compute dice coefficient between current segmentation result and groundtruth (GT)"""

    sum_GT = numpy.sum(GT)
    sum_Seg = numpy.sum(Seg)

    if (sum_GT + sum_Seg) == 0:
        dice = 1.0
    else:
        dice = (2.0 * numpy.sum(Seg[GT == 1])) / (sum_Seg + sum_GT)

    return dice


def compute_label_assignment(pred, gt):
    """Assign segmentation labels to ground truth labels using the Munkres
     algorithm.

    :param pred: A label image of the segmentation results.

    :param gt: A ground truth label image.

    :returns: ``assignment, pred_labels, gt_labels``, where ``assignment``
        is a list of ``(pred_label, gt_label)`` pairs that the munkres
        algorithm joined together, ``pred_labels`` is a list of the labels
        in the predicted label image, and ``gt_labels`` is the list
        of labels in the ground truth label image.
    """
    import munkres
    from skimage.measure import regionprops

    pred_rprops = regionprops(pred)
    gt_rprops = regionprops(gt)

    if len(pred_rprops) == 0 or len(gt_rprops) == 0:
        assignment = numpy.zeros((0, 2), dtype=numpy.int)

    else:
        # Compute the assignment score matrix (higher is more likely to be
        # assigned to each other).
        # Uses Dice coefficient to compute scores for munkres algorithm.
        D = numpy.zeros((len(pred_rprops), len(gt_rprops)), dtype=numpy.float)
        for i, rprops_seg in enumerate(pred_rprops):
            for j, rprops_gt in enumerate(gt_rprops):

                min_row_s, min_col_s, max_row_s, max_col_s = rprops_seg.bbox
                min_row_g, min_col_g, max_row_g, max_col_g = rprops_gt.bbox

                def get_overlap(a, b):
                    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

                overlap_row = get_overlap([min_row_s, max_row_s], [min_row_g, max_row_g])
                overlap_col = get_overlap([min_col_s, max_col_s], [min_col_g, max_col_g])

                if overlap_row > 0 and overlap_col > 0:
                    D[i, j] = dice(pred == rprops_seg.label, gt == rprops_gt.label)
                else:
                    D[i, j] = 0

        # init minimum assignment algorithm
        # print('\t...running Munkres... {0} pred regions, {1} gt regions'.format(len(pred_rprops), len(gt_rprops)))
        D *= -1
        m = munkres.Munkres()

        if D.shape[0] <= D.shape[1]:
            assignment = numpy.asarray(m.compute(D.copy()))
        else:
            assignment = numpy.asarray(m.compute(D.T.copy()))
            assignment = assignment[:, ::-1]

        # remove high cost assignments
        costs = numpy.asarray([D[i, j] for i, j in assignment])
        assignment = assignment[costs < 0]

        print('\tMunkres: {0} pred regions, {1} gt regions, {2} assigned'.format(len(pred_rprops), len(gt_rprops), len(assignment)))

    # prepare labels
    pred_labels = numpy.asarray([r.label for r in pred_rprops])
    gt_labels = numpy.asarray([r.label for r in gt_rprops])

    return assignment, pred_labels, gt_labels


def compute_recognition_assignment(assignment, pred_labels, gt_labels):
    """Decides for each predicted label whether it is a true or false
    positive, and for each ground truth label whether the corresponding
    object is a false negative.

    Expects the outputs of ``compute_label_assignment``.

    :returns: ``tp_idx, fp_idx, fn_idx`` such that (a) for the i-th
        predicted label, ``tp_idx`` is 1 if the given prediction is
        a true positive according to the assignment and 0 otherwise;
        analogously ``fp_idx`` will be 1 if the given prediction is
        a false positive given the assignment. In ``fn_idx``, the j-th
        item will be 1 if the j-th ground truth label did not get
        any predicted counterpart assigned to it.
    """
    pred_visited = numpy.zeros(len(pred_labels))
    gt_visited = numpy.zeros(len(gt_labels))

    for i_seg, i_gt in assignment:
        pred_visited[i_seg] = 1
        gt_visited[i_gt] = 1

    tp_idx = numpy.nonzero(pred_visited)[0]
    fp_idx = numpy.nonzero(pred_visited == 0)[0]
    fn_idx = numpy.nonzero(gt_visited == 0)[0]

    return tp_idx, fp_idx, fn_idx


##############################################################################


def load_input_images(images, labels, input_root, dtype='uint8'):
    """Loads the images for the specified labels from the input root.

    :param images: A list of image names (can be without suffix).

    :param labels: A list of the desired labels.

    :param input_root: The root labels/masks directory.

    :param dtype: The desired numpy array dtype.

    :return: A dict of dicts. Top-level keys are image names,
        lower-level keys are labels. The values in the inside dict
        are the
    """
    logging.info('Loading images...')
    _image_load_start = time.time()
    _imset = {}
    if images:
        _imset = set(images)
    else:
        raise ValueError('Cannot evaluate: no image names'
                         ' specified in --images!')

    #: The actual filenames (inside input_root/label_dir/ )
    #  of the input images. Don't confuse this with the image
    #  names that are passed through --images; they don't have
    #  the file type ending (.png).
    _input_fnames = []
    imgs_per_fname = collections.defaultdict(dict)

    #: For each fname (corresponding to an input image),
    #  store the loaded image with the given label.
    for label in labels:
        label_dir = os.path.join(input_root, label)

        for fname in os.listdir(label_dir):

            _skip = False
            if _imset:
                _skip = True
                if fname in _imset:
                    _skip = False
                elif os.path.splitext(fname)[0] in _imset:
                    _skip = False
            if _skip:
                continue

            if fname not in _input_fnames:
                _input_fnames.append(fname)

            img_path = os.path.join(label_dir, fname)
            image = imread(img_path, mode='L').astype(dtype)

            # Image normalization goes here.

            imgs_per_fname[fname][label] = image

    logging.info('Images loaded. Total: {0}\n'
                 'Time taken: {1:.2f} s'
                 ''.format(len(labels) * len(imgs_per_fname),
                           time.time() - _image_load_start))

    return imgs_per_fname


##############################################################################
# Logging

def show_detection_result(input_image,
                          tp_idx, fp_idx, fn_idx,
                          tp, fp, fn,
                          pred_centroids, gt_centroids,
                          show_tp=False,
                          channel_name=None):
    """Plots the detection result & shows it.
    """
    fp_coords = pred_centroids[fp_idx] if len(pred_centroids) > 0 else numpy.zeros((0, 2))
    fn_coords = gt_centroids[fn_idx] if len(gt_centroids) > 0 else numpy.zeros((0, 2))
    tp_coords = pred_centroids[tp_idx] if len(pred_centroids) > 0 else numpy.zeros((0, 2))

    import matplotlib.pyplot as plt

    plt.imshow(input_image, cmap=plt.cm.gray, interpolation="nearest")

    plt.plot(fp_coords[:, 1], fp_coords[:, 0], 'co', markersize=9, label="FP", alpha=0.7)
    plt.plot(fn_coords[:, 1], fn_coords[:, 0], 'mo', markersize=9, label="FN", alpha=0.7)
    if show_tp:
        plt.plot(tp_coords[:, 1], tp_coords[:, 0], 'go', markersize=5, label="TP", alpha=0.7)

    plt.legend(fontsize=18)
    if channel_name is not None:
        plt.title(("Channel: %s | FP: %d, FN: %d, TP: %d"
                   "" % (channel_name, fp, fn, tp)), fontsize=22)
    else:
        plt.title(("FP: %d, FN: %d, TP: %d" % (fp, fn, tp)), fontsize=22)

    plt.xlim([0, input_image.shape[1] - 1])
    plt.ylim([input_image.shape[0] - 1, 0])

    plt.show(block=True)


def show_prob_map(input_image, prob_map_channel):
    """Plot the probability map for a given channel."""
    from mhr.utils.colormaps import magma
    import matplotlib.pyplot as plt
    plt.imshow(prob_map_channel, cmap=magma, interpolation='nearest')
    plt.imshow(input_image, interpolation='nearest', alpha=0.2)
    plt.show(block=True)


def print_report(lr):
    """Prints out the reported results. Expects a dict of per-label
    result dicts."""
    for l, r in lr.items():
        print('Label: {0}'.format(l))
        print('\tNumber of objects: gt: {0}, de: {1}'
              ''.format(r['n_gt_labels'], r['n_pred_labels']))
        print('\tDice: {0:.3f}'
              ''.format(r['dice']))
        print('\tTP: {0}, FP: {1}, FN: {2}'
              ''.format(r['tp'], r['fp'], r['fn']))
        print('\tRec: {0:.3f}, Prec: {1:.3f}, F-sc: {2:.3f}'
              ''.format(r['rec'], r['prec'], r['fsc']))


##############################################################################
# Wrappers for more complex tasks

def compute_eval_metrics(gt_label_img, pred_label_img, gt_mask, prob_mask):
    # Get ground truth centroids & label img. for given channel
    gt_regions = regionprops(gt_label_img)
    gt_centroids = numpy.array([r.centroid for r in gt_regions])

    # Run evaluation: compute label assignment,
    # recognition assignment, and compute evaluation metrics.
    assignment, pred_label_img, gt_labels = \
        compute_label_assignment(pred_label_img, gt_label_img)

    tp_idx, fp_idx, fn_idx = \
        compute_recognition_assignment(assignment, pred_label_img, gt_labels)

    tp, fp, fn = len(tp_idx), len(fp_idx), len(fn_idx)
    if tp == 0:
        rec, prec, fsc = 0.0, 0.0, 0.0
    else:
        rec, prec = tp / (tp + fn), tp / (tp + fp)
        fsc = (2 * rec * prec) / (rec + prec)

    dice_coeff = dice(prob_mask, gt_mask)

    r = {
        'n_gt_labels': len(gt_labels), 'n_pred_labels': len(pred_label_img),
        'gt_centroids': gt_centroids,
        'tp': tp, 'fp': fp, 'fn': fn,
        'tp_idx': tp_idx, 'fp_idx': fp_idx, 'fn_idx': fn_idx,
        'rec': rec, 'prec': prec, 'fsc': fsc,
        'dice': dice_coeff,
    }
    return r


def compute_eval_metrics_multichannel(pred_label_img,
                                      gt_channels_mask,
                                      labels):
    """Gather the evaluation metrics for each output label."""
    # Extract ground truth.
    labels_results = collections.OrderedDict()
    for _ch, output_label in enumerate(labels):

        # print('\t...Computing label {0} ({1})'.format(_ch, output_label))

        ch_pred_label_img = pred_label_img[_ch]
        ch_pred_label_mask = ch_pred_label_img * 1
        ch_pred_label_mask[ch_pred_label_mask != 0] = 1

        ch_gt_mask = gt_channels_mask[_ch]
        ch_gt_label_img = label(ch_gt_mask)

        r = compute_eval_metrics(ch_gt_label_img,
                                 ch_pred_label_img,
                                 ch_gt_mask,
                                 ch_pred_label_mask)

        labels_results[output_label] = r
    return labels_results


def evaluate_detection(data_loader, detector, show_results=False,
                       subsample_window=None,
                       return_maps=False):
    """Returns a dict of per-image per-label detection dicts."""
    ##########################################################################
    # Run detection/evaluation loop.
    results = collections.OrderedDict()

    # This is only used for logging...
    _available_image_files = data_loader.dataset.get_available_image_names()

    data_iterator = iter(data_loader)

    prob_maps = []
    prob_masks = []
    label_imgs = []

    for i in range(len(data_loader)):

        ######################################################################
        # print('Validation: Processing test image: {0}'
        #       ''.format(_available_image_files[i]))
        # logging.info('Processing test image: {0}'
        #              ''.format(_available_image_files[i]))

        # Load data
        #  -- assumes the iterator has a batch size of 1
        input_batch, output_batch = next(data_iterator)

        input_channels = input_batch[0]
        output_channels = output_batch[0]

        # Subsampling greatly speeds up validation, at the cost of
        # approximating the true validation error instead of computing
        # it accurately.
        if subsample_window:
            sh, sw = subsample_window
            orig_h, orig_w = input_channels.shape[-2:]
            _nnz_margin = min(400, (orig_h - sh - 1) // 2)
            t = numpy.random.randint(_nnz_margin, orig_h - sh - _nnz_margin)
            l = numpy.random.randint(_nnz_margin, orig_w - sw - _nnz_margin)
            input_channels = input_channels[:, t:t+sh, l:l+sw]
            output_channels = output_channels[:, t:t+sh, l:l+sw]
            # print('Subsampled shapes: input channels: {0}, output channels: {1}'
            #       ''.format(input_channels.shape, output_channels.shape))

        # Run detector.
        # Instead of detector.run(), we unroll the steps, in order
        # to have the intermediate results for plotting/eval later.
        WINDOW_SIZE = (256, 512)
        proba_logit = apply_model(input_channels,
                                  detector.net,
                                  window_size=WINDOW_SIZE)
        prob_map = detector.logits2probs(proba_logit)
        prob_mask, pred_centroids, pred_label_img = \
            detector.run_detection(prob_map)

        prob_maps.append(prob_map)
        prob_masks.append(prob_mask)
        label_imgs.append(pred_label_img)

        # Evaluate for each output label
        # print('\t...Model applied, computing multichannel metrics')
        label_results = compute_eval_metrics_multichannel(pred_label_img,
                                                          output_channels,
                                                          labels=data_loader.output_labels)

        # Render results?
        if show_results:
            for _ch, output_label in enumerate(data_loader.output_labels):
                show_prob_map(input_channels.sum(axis=0),
                              prob_map_channel=prob_map[_ch])
                r = label_results[output_label]
                show_detection_result(input_channels.sum(axis=0),
                                      r['tp_idx'], r['fp_idx'], r['fn_idx'],
                                      r['tp'], r['fp'], r['fn'],
                                      pred_centroids, r['gt_centroids'],
                                      show_tp=False,
                                      channel_name=output_label)

        results[i] = label_results

    # Aggregate results
    macro_aggregated_metrics = ['tp', 'fp', 'fn', 'dice', 'prec', 'rec', 'fsc']
    micro_aggregated_metrics = ['tp', 'fp', 'fn']

    agg_results = collections.defaultdict(list)   # across images AND labels, macro-averages
    label_macro_agg_results = dict()              # across images for individual labels, macro-average
    label_micro_agg_results = dict()              # across images for individual labels, micro-average
                                                  # (only sums tp, fp, fn and computes results from that)
    for img_idx in results:
        i_results = results[img_idx]

        for label in i_results:
            il_results = i_results[label]
            if label not in label_macro_agg_results:
                label_macro_agg_results[label] = collections.defaultdict(list)
            if label not in label_micro_agg_results:
                label_micro_agg_results[label] = collections.defaultdict(list)

            for k, v in il_results.items():
                if k in macro_aggregated_metrics:
                    agg_results[k].append(v)
                    label_macro_agg_results[label][k].append(v)
                if k in micro_aggregated_metrics:
                    label_micro_agg_results[label][k].append(v)

            tp, fp, fn = il_results['tp'], il_results['fp'], il_results['fn']
            label_micro_agg_results[label]['support_f1'].append(tp + fp + fn)

    _label_micro_supports = {label: numpy.sum(label_micro_agg_results[label]['support_f1'])
                             for label in label_micro_agg_results}
    label_micro_avg_results = {
        label: {
            'tp': numpy.sum(label_micro_agg_results[label]['tp']),
            'fp': numpy.sum(label_micro_agg_results[label]['fp']),
            'fn': numpy.sum(label_micro_agg_results[label]['fn']),
        }
        for label in label_micro_agg_results
    }
    for label in label_micro_avg_results:
        _lmar = label_micro_avg_results[label]
        tp, fp, fn = _lmar['tp'], _lmar['fp'], _lmar['fn']
        rec = tp / (tp + fn)
        prec = tp / (tp + fp)
        fsc = (2 * rec * prec) / (rec + prec)
        label_micro_avg_results[label]['rec'] = rec
        label_micro_avg_results[label]['prec'] = prec
        label_micro_avg_results[label]['fsc'] = fsc

    label_macro_avg_results = {label: {k: numpy.mean(v)
                                       for k, v in label_macro_agg_results[label].items()}
                               for label in label_macro_agg_results}
    avg_results = {k: numpy.mean(v) for k, v in agg_results.items()}

    if return_maps:
        return results, label_micro_avg_results, label_macro_avg_results, avg_results, prob_maps, prob_masks, label_imgs
    return results, label_micro_avg_results, label_macro_avg_results, avg_results


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--model', required=True,
                        help='The state dict of the model that you wish to use.')
    parser.add_argument('--lasagne', action='store_true',
                        help='Indicates that the model params have been exported'
                             ' as a list from Lasagne and need to be converted'
                             ' to a PyTorch state dict.')
    parser.add_argument('--n_input_channels', type=int, default=1,
                        help='How many input channels are there. Shortcut for now;'
                             ' this should be saved together with the state dict.')
    parser.add_argument('--multiencoder_model', action='store_true',
                        help='If you are using multiple input labels, you may'
                             ' want to have a separate encoder stack for'
                             ' each input channel. This model implements'
                             ' that; the decoder portion then gets a simple'
                             ' sum of the encoded representations (incl. skip'
                             ' connections) as the input.')
    parser.add_argument('--selfattn_model', action='store_true',
                        help='Adds one self-attention hidden conv layer on top'
                             ' of the output, with 32 filters as attention "heads";'
                             ' then another 1x1 output, and returns'
                             ' the maximum of these two. The point is to try'
                             ' whether looking at the previous results'
                             ' that are already known to be results'
                             ' helps.')

    parser.add_argument('-i', '--images', nargs='+', required=True,
                        help='The list of image files that you want to process,'
                             ' relative to the --input_root + --input_labels.'
                             ' This same list will also be used to load ground'
                             ' truth *labels* (not masks!) from the directory'
                             ' --gt_labels_root + --output_label.')

    # Naming output dumps
    parser.add_argument('--testset', action='store', default='independent',
                        help='The testset name is necessary for naming the output'
                             ' dump directories properly.')
    parser.add_argument('--augmentation', action='store_true',
                        help='Saying whether augmentation was used to train'
                             ' the model is necessary for naming the output'
                             ' dump directories properly.')
    parser.add_argument('--exp_tag', action='store',
                        help='The model might have some special tag. Again,'
                             ' necessary for naming the output dump dirs'
                             ' properly.')

    parser.add_argument('--input_root', default='',
                        help='Read files from this directory. If -i is not given,'
                             ' applies the model to *all* the images in the directory.'
                             ' If -i is given, only those files will be processed.')
    parser.add_argument('--input_labels', required=True, nargs='+',
                        help='The labels with masks used as input channels. Accepts'
                             ' both a list and a comma-separated list. The order'
                             ' of the labels is the order of the corresponding'
                             ' input channels to the model.')

    parser.add_argument('--gt_labels_root', required=True,
                        help='If set, will also load label images from this'
                             ' directory and consider them ground truth for'
                             ' detection. Uses --images (-i) fnames to select'
                             ' the ground truth images, and --ouptut_label'
                             ' to select the given symbol class directory'
                             ' relative to this root.')
    parser.add_argument('--output_labels', required=True, nargs='+',
                        help='The output labels for the evaluated model. Accepts'
                             ' both a list and a comma-separated list. The order'
                             ' of the labels is the order of the corresponding'
                             ' input channels to the model.')

    parser.add_argument('--normalize_images', action='store_true',
                        help='Apply image normalization (auto-invert, intensity stretch,'
                             ' binarization).')
    # parser.add_argument('--augmentation', action='store_true',
    #                     help='Apply default augmentations on the fly.')

    parser.add_argument('--output_dir', default='',
                        help='If set without -o, will use the image file basenames'
                             ' and store them into this directory. If the directory'
                             ' does not exist, it will be created.')
    parser.add_argument('-o', '--output', nargs='+',
                        help='The corresponding list of output file basenames.'
                             ' If given, has to be of the same length as the'
                             ' list to -i. Stores the outputs with these names'
                             ' relative to --output_dir. [NOT IMPLEMENTED]')
    parser.add_argument('--output_csv',
                        help='If set, will dump the mob-deetection-paper CSV'
                             ' output to this file.')

    parser.add_argument('-z', '--zoom', action='store', type=float, default=0,
                        help='Zoom the input image by this factor before anything'
                             ' else is applied. (For example: use 0.5 to downscale'
                             ' to half the original size.')

    ###########
    # Detector params
    parser.add_argument('--detector_threshold', type=float, default=0.5,
                        help='The threshold at which the pixelwise probability'
                             ' map is thresholded.')
    parser.add_argument('--min_area', type=int, default=5,
                        help='The minimum area of a mask connected component'
                             ' that will be considered.')
    parser.add_argument('--detector_stride_ratio', type=int, default=1,
                        help='When the model is applied over windows tiled over'
                             ' the input image, they overlap proportionally to'
                             ' the stride ratio.')

    parser.add_argument('--dtype', default='uint8',
                        help='Load the input images with the given numpy dtype.')

    parser.add_argument('--show_results', action='store_true',
                        help='Render results using matplotlib.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    exp_name = build_experiment_name(args)

    logging.info('Starting main...')
    _start_time = time.time()

    # Process input_labels with comma separation
    if ',' in args.input_labels[0]:
        args.input_labels = args.input_labels[0].split(',')
        args.n_input_channels = len(args.input_labels)
    else:
        args.n_input_channels = 1

    if ',' in args.output_labels[0]:
        args.output_labels = args.output_labels[0].split(',')
        args.n_output_channels = len(args.output_labels)
    else:
        args.n_output_channels = 1

    logging.info('Preparing model...')
    logging.info('\tLoading FCN model params from state dict: {0}'.format(args.model))
    if args.lasagne:
        with open(args.model, 'rb') as hdl:
            lasagne_params = pickle.load(hdl, encoding='latin1')
        state_dict = lasagne_fcn_2_pytorch_fcn(lasagne_params)
    else:
        state_dict = torch.load(args.model)

    logging.info('\tInitializing FCN model architecture')
    model = init_model(n_input_channels=args.n_input_channels,
                       n_output_channels=args.n_output_channels,
                       multiencoder=args.multiencoder_model,
                       self_attention=args.selfattn_model)

    # if args.multiencoder_model:
    #     encoders = [FCNEncoder(n_input_channels=1) for _ in args.input_labels]
    #     decoder = FCNDecoder(n_input_channels=1)
    #     model = MultiEncoderFCN(encoders=encoders, decoder=decoder)
    # else:
    #     model = FCN(n_input_channels=args.n_input_channels,
    #                 n_output_channels=args.n_output_channels)

    logging.info('\tInitializing model weights from loaded params.')
    model.load_state_dict(state_dict)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logging.info('\tTransferring model to CUDA.')
        model.cuda()

    # Set dropout
    model.dropout11.train(mode=False)
    model.dropout14.train(mode=False)

    # Prepare prob maps dump
    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        args.output_dir = os.path.join(args.output_dir, exp_name)
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)

    ##########################################################################
    # Get detection params.
    detector_threshold = args.detector_threshold
    stride_ratio = args.detector_stride_ratio
    min_area = args.min_area
    input_root = args.input_root
    input_labels = args.input_labels
    output_labels = args.output_labels
    images = args.images
    show_results = args.show_results

    ##########################################################################
    # Prepare the detector.
    detector = ConnectedComponentDetector(net=model,
                                          is_logits=True,
                                          stride_ratio=stride_ratio,
                                          threshold=detector_threshold,
                                          min_area=min_area)

    ##########################################################################
    # Prepare data loading. Use the exclude_img_names mechanism
    # to only load images from the test set.
    available_names = os.listdir(os.path.join(input_root,
                                              input_labels[0]))
    # non_test_names = [name for name in available_names
    #                   if name not in args.images]

    dataset = MUSCIMALabelsDataset(root=input_root,
                                   include_img_names=images)

    data_loader = MUSCIMALabelIterator(dataset=dataset,
                                       train=True,
                                       shuffle=False,
                                       ensure_masks_in=True,
                                       ensure_masks_out=False,
                                       input_labels=input_labels,
                                       output_labels=output_labels,
                                       batch_size=1,
                                       sample_windows=False,
                                       prescale=args.zoom)

    # Check for loaded images
    logging.info('Dataset: includes images {0},\n\nexcludes {1}'
                 ''.format(dataset.include_img_names,
                           dataset.exclude_img_names))
    for l in input_labels + output_labels:
        logging.info('Dataset: label {0}: loaded {1} images'
                     ''.format(l, len(dataset.label_images[l])))

    detailed_results, \
    label_micro_avg_results, label_macro_avg_results, avg_results, \
    prob_maps, prob_masks, pred_label_imgs = \
        evaluate_detection(
            data_loader,
            detector,
            show_results=show_results,
            return_maps=True)

    # Rescale prob. maps, masks, and detection label imgs. to original size,
    # so that the exports provide actually useful data.
    if args.zoom != 0:
        if len(output_labels) > 1:
            raise ValueError('Cannot un-zoom detection with more than one'
                             ' output label.')
        import cv2
        # We should rather load original images here and resize to their size,
        # but for now this will do...
        unzoom = int(1. / args.zoom)
        logging.info('Unzooming with ratio {}'.format(unzoom))
        print('shapes: {}'.format([img.shape for img in pred_label_imgs]))
        prob_maps = [numpy.array([cv2.resize(img[0].astype('uint16'),
                                             dsize=None, fx=unzoom, fy=unzoom,
                                             interpolation=cv2.INTER_NEAREST)])
                     for img in prob_maps]
        prob_masks = [numpy.array([cv2.resize(img[0].astype('uint16'),
                                              dsize=None, fx=unzoom, fy=unzoom,
                                              interpolation=cv2.INTER_NEAREST)])
                      for img in prob_masks]
        pred_label_imgs = [numpy.array([cv2.resize(img[0].astype('uint16'),
                                                   dsize=None, fx=unzoom, fy=unzoom,
                                                   interpolation=cv2.INTER_NEAREST)])
                           for img in pred_label_imgs]

    ###################
    # Dump probability maps
    if args.output_dir:
        print('Dumping output to directory: {0}'.format(args.output_dir))
        prob_maps_dict = {img_name: m
                          for img_name, m in zip(dataset.include_img_names, prob_maps)}
        maps_file = os.path.join(args.output_dir, 'prob_maps.{0}.pkl'.format(exp_name))
        with open(maps_file, 'wb') as hdl:
            pickle.dump(prob_maps_dict, hdl, protocol=pickle.HIGHEST_PROTOCOL)
        masks_file = os.path.join(args.output_dir, 'prob_masks.{0}.pkl'.format(exp_name))
        with open(masks_file, 'wb') as hdl:
            pickle.dump(prob_maps_dict, hdl, protocol=pickle.HIGHEST_PROTOCOL)

    ###################
    # Dump mob-detection-paper CSV
    if args.output_csv:
        logging.info('Exporting csv: {}'.format(args.output_csv))
        if len(output_labels) > 1:
            raise ValueError('Cannot export CSV with more than one output label.')
        csv_texts = []
        for fname, l_img in zip(dataset.include_img_names, pred_label_imgs):
            clsname = output_labels[0]
            csv_text = label_img2mobcsv(l_img[0], clsname, fname)
            csv_texts.append(csv_text)
        with open(args.output_csv, 'w') as hdl:
            hdl.write('\n'.join(csv_texts) + '\n')

    ###################
    # Print detailed results
    for i, img_name in zip(detailed_results, dataset.include_img_names):
        print('Image {0}'.format(img_name))
        print_report(detailed_results[i])

    ###################
    # Average the evaluation results
    aggregate_results = {}
    for l in output_labels:
        recs = [r[l]['rec'] for r in detailed_results.values()]
        precs = [r[l]['prec'] for r in detailed_results.values()]
        fscs = [r[l]['fsc'] for r in detailed_results.values()]
        dices = [r[l]['dice'] for r in detailed_results.values()]
        aggregate_results[l] = {'rec': numpy.average(recs),
                                'prec': numpy.average(precs),
                                'fsc': numpy.average(fscs),
                                'dice': numpy.average(dices)}

    print('==========================================================')
    print('Macro-averaged results:')
    for l in aggregate_results:
        print('\nLabel: {0}'.format(l))
        print('\tRecall:\t{0:.3f}'.format(aggregate_results[l]['rec']))
        print('\tPrecision:\t{0:.3f}'.format(aggregate_results[l]['prec']))
        print('\tF-score:\t{0:.3f}'.format(aggregate_results[l]['fsc']))
        print('\tDice:\t{0:.3f}'.format(aggregate_results[l]['dice']))

    print('==========================================================')
    print('Micro-averaged results:')
    for l in label_micro_avg_results:
        print('\nLabel: {0}'.format(l))
        print('\tRecall:\t{0:.3f}'.format(label_micro_avg_results[l]['rec']))
        print('\tPrecision:\t{0:.3f}'.format(label_micro_avg_results[l]['prec']))
        print('\tF-score:\t{0:.3f}'.format(label_micro_avg_results[l]['fsc']))

    logging.info('Evaluation.py done in {0:.2f} s'
                 ''.format(time.time() - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    main(args)
