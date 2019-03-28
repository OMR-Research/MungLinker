#!/usr/bin/env python
"""This is a script that runs the object detection server.
Will be adapted for munglinker later."""
from __future__ import print_function, unicode_literals

import argparse
import copy
import logging
import os
import pickle
import pprint
import socket

from munglinker.detector import ConnectedComponentDetector

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import time
import uuid


import numpy
import skimage.measure
import cv2

from muscima.cropobject import CropObject
from muscima.io import export_cropobject_list


__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################

def connected_components2bboxes(labels):
    """Returns a dictionary of bounding boxes (upper left c., lower right c.)
    for each label.

    >>> labels = [[0, 0, 1, 1], [2, 0, 0, 1], [2, 0, 0, 0], [0, 0, 3, 3]]
    >>> bboxes = connected_components2bboxes(labels)
    >>> bboxes[0]
    [0, 0, 4, 4]
    >>> bboxes[1]
    [0, 2, 2, 4]
    >>> bboxes[2]
    [1, 0, 3, 1]
    >>> bboxes[3]
    [3, 2, 4, 4]


    :param labels: The output of cv2.connectedComponents().

    :returns: A dict indexed by labels. The values are quadruplets
        (xmin, ymin, xmax, ymax) so that the component with the given label
        lies exactly within labels[xmin:xmax, ymin:ymax].
    """
    bboxes = {}
    for x, row in enumerate(labels):
        for y, l in enumerate(row):
            if l not in bboxes:
                bboxes[l] = [x, y, x+1, y+1]
            else:
                box = bboxes[l]
                if x < box[0]:
                    box[0] = x
                elif x + 1 > box[2]:
                    box[2] = x + 1
                if y < box[1]:
                    box[1] = y
                elif y + 1 > box[3]:
                    box[3] = y + 1
    return bboxes


class DetectionOSError(OSError):
    pass


def load_pytorch_model(model_filename, n_input_channels=1, n_output_channels=1,
                       multiencoder=False, self_attention=False):
    import torch
    # from mhr.experiments.fcn.model import FCN
    from munglinker.model import init_model

    state_dict = torch.load(model_filename,
                            map_location=lambda storage, loc: storage)['model_state_dict']

    model = init_model(n_input_channels=n_input_channels,
                       n_output_channels=n_output_channels,
                       multiencoder=multiencoder,
                       self_attention=self_attention)
    model.load_state_dict(state_dict)

    use_cuda = torch.cuda.is_available()
    print('Will use CUDA: {}'.format(use_cuda))
    if use_cuda:
        model.cuda()

    return model


def load_lasagne_model(primitive,
                       params_file,
                       params_file_ch,
                       primitive_model='primitive_detector'):
    # Requires running in the (mmtt) environment!
    from omr.train import select_model
    from omr.omr_app import OpticalMusicRecognizer
    from lasagne_wrapper.network import SegmentationNetwork

    omr = OpticalMusicRecognizer(note_detector=None,
                                 system_detector=None,
                                 bar_detector=None)

    # init mask hull net
    model = select_model(primitive_model)
    net = model.build_model()
    detector = SegmentationNetwork(net)
    detector.load(params_file)

    # init convex hull net
    net_ch = model.build_model()
    detector_ch = SegmentationNetwork(net_ch)
    detector_ch.load(params_file_ch)

    omr.add_primitives_detector(primitive, detector, detector_ch)

    return omr


def reparametrize_lasagne_model(primitive,
                                params_file,
                                params_file_ch,
                                omr):
    """Loads new params into the model."""
    if len(omr.primitive_detector) == 0:
        omr = load_lasagne_model(primitive, params_file, params_file_ch)
        return omr

    # Pick an old primitive
    old_primitive = list(omr.primitive_detector.keys())[0]

    detector = omr.primitive_detector[old_primitive]
    detector.load(params_file)

    # omr.primitive_detector[primitive] = detector
    # omr.primitive_detector[old_primitive] = None

    detector_ch = omr.primitive_detector_ch[old_primitive]
    detector_ch.load(params_file_ch)

    omr.add_primitives_detector(primitive, detector=detector, detector_ch=detector_ch)
    # omr.primitive_detector_ch[primitive] = detector_ch
    # omr.primitive_detector_ch[old_primitive] = None

    del omr.primitive_detector_ch[old_primitive]
    del omr.primitive_channel_mapping[old_primitive]
    del omr.primitive_detector[old_primitive]

    return omr


class DetectionWrapper:
    """Currently only works for single-channel output models."""
    def __init__(self, model_dir, output_label,  input_labels=['fulls'],
                 detection_threshold=0.5, min_bbox_area=10,
                 start_objid=0,
                 track_objid=True,
                 model=None,
                 detector=None):

        self.model_dir = model_dir
        self.output_label = output_label
        self.input_labels = input_labels

        self.model = model   # This is here for LasagneWrapper build/reparametrize decision
        if self.model is None:
            self.model = self.load_model(model_dir, input_labels, output_label)

        # Server-side filtering
        self.min_bbox_area = min_bbox_area
        self.detection_threshold = detection_threshold

        self.detector = detector
        if self.detector is None:
            detector_kwargs = {'threshold': self.detection_threshold,
                               'min_area': self.min_bbox_area}
            self.detector = self.load_detector(model_dir,
                                               input_labels,
                                               output_label,
                                               detector_kwargs)

        self.start_objid = start_objid
        self.track_objid = track_objid


    @staticmethod
    def load_model(model_dir, input_labels, output_label):
        model_filename = DetectionWrapper.get_pytorch_params_filename(input_labels,
                                                                      model_dir,
                                                                      output_label)

        return load_pytorch_model(model_filename,
                                  n_input_channels=len(input_labels),
                                  n_output_channels=1)

    @staticmethod
    def load_detector(model_dir, input_labels, output_label,
                      detector_kwargs, detector_class=ConnectedComponentDetector):
        """Loads a fcnomr package Detector instead of just the net.

        (This is a part of the effort to refactor the DetectionWrapper
        to use the Detectors instead of re-implementing the wheel.
        The DetectionWrapper will just add the MuNG-ification mechanism.)

        The detector keyword arguments must be supplied as a dict.
        E.g., for the Connected Component Detector, the default
        kwargs would be ``{'threshold': 0.5, 'min_area': 2}``.

        Note that if the self.load_cc_detector() *instance* method
        is called it fills in some details based on the detection wrapper
        instance parameters; however, this is just a shortcut and will be
        deprecated.
        """
        net = DetectionWrapper.load_model(model_dir=model_dir,
                                          input_labels=input_labels,
                                          output_label=output_label)
        is_logits = (not net.apply_sigmoid)
        detector = detector_class(net=net, is_logits=is_logits,
                                  **detector_kwargs)
        return detector


    @staticmethod
    def get_pytorch_params_filename(input_labels, model_dir, output_label):
        if not os.path.isdir(model_dir):
            raise DetectionOSError('Model directory {0} not found!'.format(model_dir))
        # Model file name is generated automatically
        model_basename = '+'.join(input_labels) + '___' + output_label + '.tsd'
        model_filename = os.path.join(model_dir, model_basename)
        if not os.path.isfile(model_filename):
            raise DetectionOSError('Model file {0} not found!'.format(model_filename))
        return model_filename

    def run(self, image):
        """Runs a detector. Outputs the resulting CropObjects."""
        if len(image.shape) == 2:
            # Add channel dimension
            image = numpy.array([image])

        # --- From here...

        centroids, label_map = self.detector.run(image, return_labels=True)
        if label_map.ndim == 3:
            if label_map.shape[0] == 1:
                label_map = label_map[0]

        bboxes = connected_components2bboxes(label_map)
        bboxes = {i: bbox for i, bbox in bboxes.items() if i != 0}

        # segmentation_map = self.run_segmentation(image)
        # segmentation_mask = self.segmentation_map_to_mask(segmentation_map,
        #                                                   original_image=image)
        #
        # # Now: from segmentation to detection!
        # # Note that the run_detection() method discards the bounding box for label 0.
        # label_map, bboxes = self.run_detection(segmentation_mask,
        #                                        original_image=image)

        # --- Up to here, this is handled by the Detector.
        # From here, it is the job of the DetectionWrapper: converting
        # the raw detection output into MuNG.

        # ...and create CropObjects.
        cropobject_masks = self.build_cropobject_masks(label_map, bboxes)
        cropobjects = self.build_cropobjects(bboxes, cropobject_masks)

        filtered_cropobjects = self.apply_filters(cropobjects)

        if self.track_objid:
            self.start_objid += len(filtered_cropobjects)

        return filtered_cropobjects

    def build_cropobjects(self, bboxes, cropobject_masks):
        cropobjects = []
        _next_objid = self.start_objid
        for bbox_idx, mask in zip(bboxes, cropobject_masks):
            t, l, b, r = bboxes[bbox_idx]
            h = b - t
            w = r - l
            clsname = self.output_label
            cropobject = CropObject(objid=_next_objid,
                                    clsname=clsname,
                                    top=t,
                                    left=l,
                                    width=w,
                                    height=h,
                                    mask=mask)
            cropobjects.append(cropobject)
            _next_objid += 1
        return cropobjects

    def build_cropobject_masks(self, label_map, bboxes):
        """Extracts a list of CropObject masks for the detected bounding boxes.
        The mask dtype is uint8."""
        cropobject_masks = []
        for l_idx in bboxes:
            t, l, b, r = bboxes[l_idx]
            view = label_map[t:b, l:r]
            mask = numpy.zeros((b - t, r - l), dtype='uint8')
            mask[view == l_idx] = 1
            cropobject_masks.append(mask)
        return cropobject_masks

    def apply_filters(self, cropobjects):
        filtered_cropobjects = cropobjects
        filtered_cropobjects = [c for c in filtered_cropobjects
                                if (c.height * c.width) > self.min_bbox_area]
        return filtered_cropobjects

    def run_segmentation(self, image):
        """Deprecated: using self.detector.run() instead"""
        from munglinker.model import apply_model
        # from mhr.experiments.fcn.run import apply_model
        return apply_model(image, self.model)

    def run_detection(self, segmentation_mask, original_image=None):
        """Deprecated: using self.detector.run() instead"""
        # Just get connected components: masks and bounding boxes.
        labels = skimage.measure.label(segmentation_mask, background=0)
        bboxes = connected_components2bboxes(labels)
        bboxes = {i: bbox for i, bbox in bboxes.items() if i != 0}
        return labels, bboxes

    def segmentation_map_to_mask(self, segmentation_map, original_image=None):
        """Apply sigmoid and cut at 0.5. Return dtype is float32.

        If the original image is supplied, it returns an intersection
        of the thresholded detection probability map with the image foreground
        (zero is considered a background pixel in the original image).

        Deprecated: using self.detector.run() instead
        """
        segmentation_sigmoid = 1. / (1 + numpy.exp(-1 * segmentation_map))
        segmentation_mask = numpy.zeros(segmentation_sigmoid.shape, dtype='float32')
        segmentation_mask[segmentation_sigmoid > self.detection_threshold] = 1.0

        if original_image is not None:
            if len(original_image.shape) == 2:
                orig_mask = original_image[numpy.newaxis, :, :]
            elif original_image.ndim == 3:
                orig_mask = numpy.sum(original_image, axis=0)[numpy.newaxis, :, :]
                orig_mask[orig_mask != 0] = 1
            else:
                raise ValueError('Cannot deal with original image of dimension {};'
                                 ' has to be either 2-D or 3-D (channels first).'
                                 ''.format(original_image.ndim))

            logging.debug('SERVER: segmentation_mask.shape = {}, orig_mask.shape = {}'
                          ''.format(segmentation_mask.shape, orig_mask.shape))
            segmentation_mask[orig_mask == 0] = 0.0

        if segmentation_mask.ndim == 3:
            logging.warning('SERVER: segmentation mask downsized to 2-dim, had {}'
                            ' channels. (This is OK if it had 1 channel.)'
                            ''.format(segmentation_mask.shape[0]))
            segmentation_mask = segmentation_mask[0, :, :]

        return segmentation_mask


class LasagneDetectionWrapper(DetectionWrapper):
    """This class wrapps Matthias's Lasagne-based detectors.

    Note that the theano code might take ages to load,
    so it should be persisted between server runs.

    Also, instead of loading models for different classes,
    it just reparametrizes the same network.
    """
    CHULL_SYMBOLS = ['c-clef',
                     'g-clef',
                     'f-clef',
                     '8th_rest',
                     'natural']

    @staticmethod
    def load_model(model_dir, input_labels, output_label):
        """Expects model_dir to be FCNs/models/lasagne-omr/primitive_detector_orig"""
        params_file, params_file_ch = LasagneDetectionWrapper.get_lasagne_params_filenames(
            model_dir, output_label)

        return load_lasagne_model(primitive=output_label,
                                  params_file=params_file,
                                  params_file_ch=params_file_ch)

    @staticmethod
    def get_lasagne_params_filenames(model_dir, output_label):
        if not os.path.isdir(model_dir):
            raise DetectionOSError('Model directory {0} not found!'.format(model_dir))

        # Model file name is generated automatically
        params_file = os.path.join(model_dir, 'params_indep_{}.pkl'.format(output_label))
        if not os.path.isfile(params_file):
            raise DetectionOSError('Model params {0} not found!'.format(params_file))

        params_file_ch = os.path.join(model_dir, 'params_indep_{}_chull.pkl'.format(output_label))
        if not os.path.isfile(params_file_ch):
            raise DetectionOSError('Model params {0} not found!'.format(params_file_ch))

        return params_file, params_file_ch

    def run(self, image):

        # Preprocessing here!
        image = self.prepare_image(image)

        if image.ndim != 2:
            if (image.ndim == 3) and (image.shape[0]) == 1:
                image = image[0]
            else:
                raise ValueError('Cannot run Lasagne detector on other than'
                                 ' 2-D or single-channel 3-D (C,H,W) images.')

        detector_style = 'mask'
        if self.output_label in self.CHULL_SYMBOLS:
            detector_style = 'conv_hull'

        centroids, labels = self.model.detect_primitives(image,
                                                         self.output_label,
                                                         threshold_abs=0.1,
                                                         verbose=False,
                                                         detector=detector_style,
                                                         return_labels=True)

        label_map, bboxes = self.run_detection(labels, original_image=image)
        # ...and create CropObjects.
        cropobject_masks = self.build_cropobject_masks(label_map, bboxes)
        cropobjects = self.build_cropobjects(bboxes, cropobject_masks)
        filtered_cropobjects = self.apply_filters(cropobjects)
        postprocessed_cropobjects = self.apply_postprocessing(filtered_cropobjects)

        return postprocessed_cropobjects

    def run_detection(self, labels, original_image=None):
        # Just get connected components: masks and bounding boxes.
        # The labels are already returned by the Lasagne OMR class.
        bboxes = connected_components2bboxes(labels)
        bboxes = {i: bbox for i, bbox in bboxes.items() if i != 0}
        return labels, bboxes

    def apply_postprocessing(self, cropobjects):
        """Scale the cropobjects back to 2x the size"""
        def postprocess_cropobject(c):
            # Scale bounding box
            c.x *= 2
            c.y *= 2
            c.height *= 2
            c.width *= 2
            mask = cv2.resize(c.mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            c.set_mask(mask)
            return c
        return [postprocess_cropobject(c) for c in cropobjects]

    def prepare_image(self, image, zoom=0.5):
        """Normalize to 0-1 and rescale to zoom (0.5 by default, which
        is what we have trained models for)."""
        image = image.astype(numpy.float32)
        if image.max() != 0:
            image /= image.max()

        if zoom:
            new_shape = (int(image.shape[1] * zoom),
                         int(image.shape[0] * zoom))
            image = cv2.resize(image, new_shape)

        return image


class DetectionWrapperContainer(object):
    """Hides the mechanics of sharing the Lasagne models."""
    def __init__(self, model_dir, lasagne=False):

        self.model_dir = model_dir
        self.lasagne = lasagne

        self.lasagne_wrappers = {}
        self.pytorch_wrappers = {}

    def add_primitive_wrapper(self, primitive, lasagne=False, **load_kwargs):
        if lasagne:
            self.ensure_lasagne_detection_wrapper(primitive, **load_kwargs)
        else:
            self.add_pytorch_detection_wrapper(primitive, **load_kwargs)

    def ensure_lasagne_detection_wrapper(self, primitive,
                                         **wrapper_init_kwargs):

        params_file, params_file_ch = self.get_lasagne_params_filename(primitive)

        # Cheat: only load it *once*; when requesting a new class,
        # just reparametrize in get_lasagne_detection_wrapper().
        if len(self.lasagne_wrappers) == 0:

            omr = load_lasagne_model(primitive,
                                     params_file,
                                     params_file_ch)
            wrapper = LasagneDetectionWrapper(model_dir=self.model_dir,
                                              output_label=primitive,
                                              model=omr,
                                              **wrapper_init_kwargs)
            self.lasagne_wrappers[primitive] = wrapper

        elif primitive not in self.lasagne_wrappers:
            # There should only ever be one
            old_wrapper = list(self.lasagne_wrappers.values())[0]
            old_primitive = old_wrapper.output_label
            omr = old_wrapper.model
            omr = reparametrize_lasagne_model(primitive,
                                              params_file,
                                              params_file_ch,
                                              omr)
            wrapper = LasagneDetectionWrapper(model_dir=old_wrapper.model_dir,
                                              output_label=primitive,
                                              input_labels=old_wrapper.input_labels,
                                              detection_threshold=old_wrapper.detection_threshold,
                                              min_bbox_area=old_wrapper.min_bbox_area,
                                              start_objid=old_wrapper.start_objid,
                                              track_objid=old_wrapper.track_objid,
                                              model=omr)

            self.lasagne_wrappers[old_primitive] = None
            del self.lasagne_wrappers[old_primitive]

            self.lasagne_wrappers[primitive] = wrapper

    def add_pytorch_detection_wrapper(self, primitive,
                                      **wrapper_init_kwargs):

        model_filename = self.get_pytorch_params_filename(primitive)
        model = load_pytorch_model(model_filename,
                                   n_input_channels=1,
                                   n_output_channels=1)

        other_wrappers = list(self.pytorch_wrappers.values())
        if len(other_wrappers) == 0:
            wrapper = DetectionWrapper(model_dir=self.model_dir,
                                       output_label=primitive,
                                       model=model,
                                       **wrapper_init_kwargs)
        else:
            old_wrapper = other_wrappers[0]
            wrapper = DetectionWrapper(model_dir=old_wrapper.model_dir,
                                       output_label=primitive,
                                       input_labels=old_wrapper.input_labels,
                                       detection_threshold=old_wrapper.detection_threshold,
                                       min_bbox_area=old_wrapper.min_bbox_area,
                                       start_objid=old_wrapper.start_objid,
                                       track_objid=old_wrapper.track_objid,
                                       model=model)

        self.pytorch_wrappers[primitive] = wrapper

    def get_detection_wrapper(self, primitive, lasagne=False, **load_kwargs):
        if lasagne:
            return self.get_lasagne_detection_wrapper(primitive, **load_kwargs)
        else:
            return self.get_pytorch_detection_wrapper(primitive, **load_kwargs)

    def get_lasagne_detection_wrapper(self, primitive, **load_kwargs):
        self.ensure_lasagne_detection_wrapper(primitive, **load_kwargs)

        # TODO: Decide whether to use convex hull or not
        return self.lasagne_wrappers[primitive]

    def get_pytorch_detection_wrapper(self, primitive, **wrapper_init_kwargs):
        if primitive not in self.pytorch_wrappers:
            self.add_primitive_wrapper(primitive, lasagne=False,
                                       **wrapper_init_kwargs)
        return self.pytorch_wrappers[primitive]

    def get_params_filename(self, primitive, lasagne=False):
        if lasagne:
            return self.get_lasagne_params_filename(primitive)
        else:
            return self.get_pytorch_params_filename(primitive)

    def get_pytorch_params_filename(self, primitive):
        return DetectionWrapper.get_pytorch_params_filename(
            input_labels=['fulls'],
            model_dir=self.model_dir,
            output_label=primitive)

    def get_lasagne_params_filename(self, primitive):
        return LasagneDetectionWrapper.get_lasagne_params_filenames(
            model_dir=self.model_dir,
            output_label=primitive)


##############################################################################


def list_models_available_for_clsnames(clsnames, model_dir,
                                       default_input_labels=('fulls',)):
    """Searches for all available models for any subsets of the clsnames.

    Assumes the files follow the naming convention::

        input_label1+input_label2+input_label3___output_label1+output_label2.tsd

    :param clsnames: List of labels we're interested in, both as inputs
        and outputs.

    :param model_dir: Directory where the models are stored.

    :param default_input_labels: Assume those labels are also a part
        of the set of input labels.

    :returns: dict that maps each model file name (relative to ``model_dir``)
        to a tuple of ``(input_labels, output_label)``. All the output labels
        for any returned model must be part of the ``clsnames`` list; the input
        labels may also include labels from the ``default_input_labels`` list.
    """
    model_files = os.listdir(model_dir)
    model_classes = []
    for mf in model_files:
        inputs_str, outputs_str = os.path.basename(mf).split('___')
        input_labels = inputs_str.split('+')
        output_labels = outputs_str.split('+')
        model_classes.append((input_labels, output_labels))

    # Filter out those models that use different classes
    # than the requested ones.
    models2classes = dict()
    for mf, mc in zip(model_files, model_classes):
        if len([l for l in input_labels
                if (l not in clsnames)
                and (l not in default_input_labels)]) == 0:
            if len([l for l in output_labels if l not in clsnames]) == 0:
                models2classes[mf] = mc

    return models2classes


def build_model_call_chain(models2classes, initial_input_labels=('fulls',)):
    """Given the available models and the classes they use,
    orders them so that when calling them in the given order, one always
    has all its inputs available, either from the initial input,
    or from the outputs of the previous models.

    :returns: A list of model names in the order of the call chain.
        Returns only the models that are reachable
        from the initial_input_labels.
    """
    available_labels = set(initial_input_labels)
    last_iter_available_labels = None
    reachable_models = []
    made_available_by = {l: '___initial' for l in initial_input_labels}
    dependencies_per_model = dict()

    while available_labels != last_iter_available_labels:
        last_iter_available_labels = copy.deepcopy(available_labels)
        for m in models2classes:
            if m in reachable_models:
                continue
            m_in, m_out = models2classes[m]
            if set(m_in) <= available_labels:
                for newly_available_label in set(m_out).difference(available_labels):
                    made_available_by[newly_available_label] = m
                dependencies_per_model[m] = [made_available_by[l] for l in m_in]
                available_labels = available_labels.union(set(m_out))
                reachable_models.append(m)

    # Now that all remaining models are reachable,
    # we can build backpointers.
    from networkx import DiGraph
    from networkx.algorithms import topological_sort, is_directed_acyclic_graph
    g = DiGraph()
    for m in dependencies_per_model:
        for n in dependencies_per_model[m]:
            if n != '___initial':
                g.add_edge(n, m)
    if not is_directed_acyclic_graph(g):
        raise ValueError('Model mutual dependencies do not form acyclic '
                         'graph...? Models2classes: {0}'.format(pprint.pformat(models2classes)))
    call_chain = topological_sort(g)
    return call_chain


##############################################################################


BUFFER_SIZE = 1024


def receive_raw_data(conn, temp_fname):
    logging.info('Receiving data to {0}'.format(temp_fname))
    _total_received = 0
    with open(temp_fname, 'wb') as fh:
        while True:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                logging.info('Receiving finished! {0} total'.format(_total_received))
                break
            fh.write(data)
            _total_received += BUFFER_SIZE


def load_request_from_raw_data(fname):
    """Use the right reading tool for recovering the sent data.
    We assume the image is sent over as a png file."""
    logging.info('Loading request from raw data')
    with open(fname, 'rb') as fh:
        try:
            request = pickle.load(fh, encoding='latin1') #!!!!!!!!!!
        except TypeError:
            request = pickle.load(fh)   # For Py2.7: pickle.load() has no encoding() argument
    uf_request = unformat_request(request)
    return uf_request


def unformat_request(request):
    uf_request = dict()
    for k in request:
        if k == 'image???':
            # Pickle-within-pickle
            uf_request[k] = unformat_request_image(request[k])
        else:
            uf_request[k] = request[k]
    return uf_request


def unformat_request_image(f_image):
    return pickle.loads(f_image)


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-p', '--port', type=int, action='store', default=33554,
                        help='The target port of the connection.')
    parser.add_argument('-n', '--hostname', action='store', default='127.0.0.1',
                        help='The hostname ')
    parser.add_argument('-t', '--tmp_dir', action='store', required=True,
                        help='A writeable directory where the temporary files'
                             ' will be stored.')
    parser.add_argument('--n_connections', type=int, default=2, action='store',
                        help='The number of connections the server will stay up for.')

    parser.add_argument('-m', '--model_dir', action='store', required=True,
                        help='The directory where the segmentation network'
                             ' state dicts are stored. For now; assumes'
                             ' the input label is "fulls", and that the output'
                             ' label is the same as the request ``clsname``'
                             ' value; automatically constructs the model'
                             ' filename inside this directory by joining'
                             ' the input and output labels using three underscores.')

    parser.add_argument('--lasagne', action='store_true',
                        help='If set, will expect LasagneWrapper.SegmentationNetwork'
                             ' models.')

    parser.add_argument('--detection_threshold', action='store', type=float, default=0.5,
                        help='The probability that is interpreted as the presence'
                             ' of a given symbol class for that pixel when going'
                             ' from the segmentation probability maps to detection'
                             ' masks. This is an important parameter.')
    parser.add_argument('--track_objid', action='store_true',
                        help='Rely on objid increments in persistent detection'
                             ' wrappers.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.time()

    # Open socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = args.hostname  # socket.gethostname()
    s.bind((host, args.port))
    s.listen(args.n_connections)

    logging.info('Server listening for {0} connections on host {1}, port {2}...'
                 ''.format(args.n_connections, host, args.port))

    n_connections_remaining = args.n_connections

    detection_wrapper_container = DetectionWrapperContainer(model_dir=args.model_dir)

    while True:
        try:
            logging.info('Waiting for connection...')
            conn, addr = s.accept()
            logging.info('Got connection from {0}'.format(addr))

            temp_basename = 'OMRserver_in' + str(uuid.uuid4()) + '.pkl'
            temp_input_file = os.path.join(args.tmp_dir, temp_basename)

            receive_raw_data(conn, temp_fname=temp_input_file)
            request = load_request_from_raw_data(temp_input_file)

            logging.info('Received request with keys: {0}'.format(request.keys()))

            image = request['image']
            classes = request['clsname']
            logging.info('Server received image of size: {0}'.format(image.shape))
            logging.info('Server received request to detect classes: {0}'
                         ''.format(classes))

            cropobjects = []
            _next_objid = 0

            # In order to implement the cascading detection algorithm,
            # we need to rewrite this part:
            #  - order the models appropriately,
            #  - in their order:
            #       - collect the input images (this needs to be re-written in the DetectionWrapper),
            #       - run detection,
            #       - record detection results (CropObjects),
            #       - record model outputs (prob. mask? detection mask?) for further models
            #
            # On the input side, we also need to wrap up all the CropObjects in the requested
            # detection region and send them as *inputs*. (Best sent as XML, unwrapped, and
            # have the masks generated on the fly on the server side, to limit transfer volume.
            # Or, possibly, fake-merge everything from a given class into one object client-side,
            # if only the mask is required.)
            #

            for clsname in classes:
                try:

                    detector = detection_wrapper_container.get_detection_wrapper(
                        primitive=clsname,
                        lasagne=args.lasagne,
                        input_labels=['fulls'],
                        detection_threshold=args.detection_threshold,
                        start_objid=_next_objid,
                        track_objid=args.track_objid
                    )

                except DetectionOSError as e:
                    # No detector for given class
                    logging.warning('DetectionOSError for class {0}: {1}'
                                    ''.format(clsname, e.message))
                    cls_cropobjects = []

                else:
                    detector.start_objid = _next_objid
                    cls_cropobjects = detector.run(image)

                cropobjects += cls_cropobjects

                logging.info('Detected {0} cropobjects of class {1}'
                             ''.format(len(cls_cropobjects), clsname))
                if len(cropobjects) > 0:
                    _next_objid = max([c.objid for c in cropobjects]) + 1

            # Serialize the data
            cropobjects_string = export_cropobject_list(cropobjects)
            logging.debug('Cropobjects string: {0}'.format(cropobjects_string))
            cropobjects_buffer = StringIO()
            cropobjects_buffer.write(cropobjects_string + '\n')
            cropobjects_buffer.flush()
            cropobjects_buffer.seek(0)

            l = bytearray(cropobjects_buffer.read(1024), encoding='latin1')
            logging.debug('First sending: {0}'.format(l))
            while (l):
                conn.send(l)
                logging.debug('sent ', repr(l))
                l = bytearray(cropobjects_buffer.read(1024), encoding='latin-1')

            logging.info('Done sending')

        except:
            raise

        finally:
            conn.close()

    _end_time = time.time()
    logging.info('server.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
