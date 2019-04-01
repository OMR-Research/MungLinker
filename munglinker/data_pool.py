#!/usr/bin/env python
"""This is a script that..."""
from __future__ import print_function, unicode_literals

import argparse
import copy
import logging
import os
import random
import time
from glob import glob
from typing import List

import numpy as np
import yaml
from PIL import Image
from muscima.cropobject import cropobject_distance, bbox_intersection, CropObject
from muscima.grammar import DependencyGrammar
from muscima.graph import NotationGraph
from muscima.inference_engine_constants import _CONST
from muscima.io import parse_cropobject_list
from tqdm import tqdm

from munglinker.utils import config2data_pool_dict, load_grammar

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."

##############################################################################
# This should all be in a config file.

# Data point sampling parameters
THRESHOLD_NEGATIVE_DISTANCE = 200
MAX_NEGATIVE_EXAMPLES_PER_OBJECT = None
RESAMPLE_EACH_EPOCH = False

# The size of the image patches that are ouptut
PATCH_HEIGHT = 256
PATCH_WIDTH = 512
# If this is set, the image in the patch will be masked to 0,
# so the system will only have the from/to masks to work with.
PATCH_NO_IMAGE = 0

# The rescaling factor that is applied before the patch is extracted
# (In effect, setting this to 0.5 downscales by a factor of 2, so
#  the effective window w.r.t. the input image will be twice the specified
#  PATCH_HEIGHT, PATCH_WIDTH.)
IMAGE_ZOOM = 1.0

# Randomly moves the patch this many pixels away from the midpoint
# between the two sampled objects.
MAX_PATCH_DISPLACEMENT = 0

##############################################################################
# These functions are concerned with data point extraction for parser training.
# Actually, it could be wrapped by a DataPool.

distances_cache_per_file = {}


##############################################################################

class MunglinkerDataError(ValueError):
    pass


class PairwiseMungoDataPool(object):
    """This class implements the basic data pool for munglinker experiments
    that outputs just pairs of MuNG nodes from the same document. Using this
    pool means that your preparation function will have to deal with everything
    else, like having at its disposal also the appropriate image from which
    to get the input patch, if need be in your model.

    It is entirely sufficient for training the baseline decision trees without
    complications, though.
    """

    def __init__(self, mungs: List[NotationGraph],
                 images: List[np.ndarray],
                 max_edge_length=THRESHOLD_NEGATIVE_DISTANCE,
                 max_negative_samples=MAX_NEGATIVE_EXAMPLES_PER_OBJECT,
                 resample_train_entities=False,
                 grammar: DependencyGrammar = None,
                 patch_size=(PATCH_HEIGHT, PATCH_WIDTH),
                 patch_no_image=PATCH_NO_IMAGE,
                 zoom: float = IMAGE_ZOOM):
        """Initialize the data pool.

        :param mungs: The NotationGraph objects for each document
            in the dataset.

        :param images: The corresponding images of the MuNGs. If
            not provided, binary masks will be generated as a union
            of all the MuNGos' masks.

        :param max_edge_length: The longest allowed edge length, measured
            as the minimum distance of the bounding boxes of the mungo
            pair in question.

        :param max_negative_samples: The maximum number of mungos sampled
            as negative examples per mungo.

        :param resample_train_entities: If set, will re-run training entity
            sampling after the pool is exhausted. Intended to be used in
            combination with a lower number of negative samples per positive
            one, so that the network sees a lot of different negative
            samples while you still have control over the positive/negative
            sample balance.

        :param patch_size: What the size of the extracted patch should
            be (after applying zoom), specified as ``(rows, columns)``.

        :param patch_no_image: Do *NOT* use the underlying image.
            Instead, channel 0 is set to all 0s. This is one of the baselines
            we use.

        :param zoom: The rescaling factor. Setting this to 0.5 means
            the image will be downscaled to half the height & width
            before the patch is extracted.

        """
        self.mungs = mungs
        self.images = images

        self.max_edge_length = max_edge_length
        self.max_negative_samples = max_negative_samples

        self.resample_train_entities = resample_train_entities

        self.patch_size = patch_size
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]
        self.patch_no_image = patch_no_image

        self.zoom = zoom
        if self.zoom != 1.0:
            self.__zoom_images()
            self.__zoom_mungs()

        self.grammar = grammar

        self.shape = None
        self.prepare_train_entities()
        self.reset_batch_generator(ignore_resample=True)

        logging.info('Data pool prepared with shape {}'.format(self.shape))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if key.__class__ == int:
            key = slice(key, key + 1)

        batch_entities = self.train_entities[key]

        # Return the patches, targets, and the MuNGos themselves
        patches_batch = np.zeros((len(batch_entities), 3,
                                  self.patch_height, self.patch_width))
        targets = np.zeros(len(batch_entities))
        mungos_from = []
        mungos_to = []
        for i_entity, (i_image, i_mungo_pair) in enumerate(batch_entities):
            m_from, m_to = self._mungo_pair_map[i_mungo_pair]
            mungos_from.append(m_from)
            mungos_to.append(m_to)
            patch = self.load_patch(i_image, m_from, m_to)
            patches_batch[i_entity] = patch

            if m_to.objid in m_from.outlinks:
                targets[i_entity] = 1

            logging.debug('DataPool: Y: {}\tFROM: {}/{}, TO: {}/{}'
                          ''.format(int(targets[i_entity]),
                                    m_from.objid, m_from.clsname,
                                    m_to.objid, m_to.clsname))

        return [mungos_from, mungos_to, patches_batch, targets]

    def __zoom_images(self):
        images_zoomed = []
        import cv2
        for image in self.images:
            img_copy = image * 1.0
            img_zoomed = cv2.resize(img_copy, dsize=None,
                                    fx=self.zoom, fy=self.zoom).astype(image.dtype)
            images_zoomed.append(img_zoomed)
        self.images = images_zoomed

    def __zoom_mungs(self):
        if self.zoom is None:
            return
        if self.zoom == 1.0:
            return
        for mung in self.mungs:
            for m in mung.cropobjects:
                m.scale(zoom=self.zoom)

    def reset_batch_generator(self, ignore_resample=False):
        """Reset data pool with new random reordering of ``train_entities``.
        """
        if self.resample_train_entities and not ignore_resample:
            logging.info('Resampling data pool.')
            self.prepare_train_entities()

            # Shuffling the train entities shuffles the order of indices
            # into self._mungo_pair_map that contains the actual MuNGOs,
            # and at the same time preserves the pairing of the MuNGo pairs
            # to the respective images.
            permutation = [int(i) for i in np.random.permutation(len(self.train_entities))]
            shuffled_train_entities = [self.train_entities[idx] for idx in permutation]
            self.train_entities = shuffled_train_entities

    def prepare_train_entities(self):
        """Extract the triplets.
        Extract MuNGo list that the train_entities will then refer to.

        The triplets will be represented as ``(i_image, m_from, m_to)``,
        where ``i_image`` points to the originating image and ``m_from``
        and ``m_to`` are one instance of sampled mungo pairs.
        """
        self.train_entities = []
        self._mungo_pair_map = []
        n_entities = 0
        for i_doc, mung in enumerate(tqdm(self.mungs, desc="Loading MuNG-pairs")):
            object_pairs = PairwiseMungoDataPool.get_object_pairs(
                mung.cropobjects,
                max_object_distance=self.max_edge_length,
                max_negative_samples=self.max_negative_samples,
                grammar=self.grammar)
            for (m_from, m_to) in object_pairs:
                # Try extracting a target patch. If this fails, don't add
                # the entity.
                try:
                    self.load_patch(i_doc, m_from, m_to)
                except MunglinkerDataError:
                    logging.info('Object pair {} --> {} does not fit within patch; skipped.'
                                 ''.format(m_from.uid, m_to.uid))
                    continue

                self._mungo_pair_map.append((m_from, m_to))
                self.train_entities.append([i_doc, n_entities])
                n_entities += 1

        # n_items x n_outputs x
        self.shape = [len(self.train_entities)]

    def load_patch(self, i_image, mungo_from, mungo_to):
        image = self.images[i_image]
        patch = self.get_x_patch(image, mungo_from, mungo_to)
        return patch

    def get_x_patch(self, image, mungo_from, mungo_to):
        """
        Assumes image is larger than patch.

        :param image:
        :param mungo_from:
        :param mungo_to:

        :return: A 3 * patch_height * patch_width array. Channel 0
            is the input image, channel 1 is the from-mungo mask,
            channel 2 is the to-mungo mask.
        """
        m_vert, m_horz = self.__compute_patch_center(mungo_from, mungo_to)
        patch_radius_v = self.patch_height // 2
        patch_radius_h = self.patch_width // 2
        t, l, b, r = m_vert - patch_radius_v, \
                     m_horz - patch_radius_h, \
                     m_vert + patch_radius_v, \
                     m_horz + patch_radius_h
        bbox_patch = t, l, b, r

        output = np.zeros((3, (b - t), (r - l)))
        bbox_image = 0, 0, image.shape[0], image.shape[1]
        bbox_of_image_wrt_patch = bbox_intersection(bbox_image, bbox_patch)
        i_crop_t, i_crop_l, i_crop_b, i_crop_r = bbox_of_image_wrt_patch
        image_crop = image[i_crop_t:i_crop_b, i_crop_l:i_crop_r]

        bbox_of_patch_wrt_image = bbox_intersection(bbox_patch, bbox_image)
        i_patch_t, i_patch_l, i_patch_b, i_patch_r = bbox_of_patch_wrt_image

        try:
            if not self.patch_no_image:
                output[0][i_patch_t:i_patch_b, i_patch_l:i_patch_r] = image_crop
        except ValueError as e:
            print('Image shape: {}'.format(image.shape))
            print('Patch bbox:  {}'.format(bbox_patch))
            print('bbox_of_image_wrt_patch: {}'.format(bbox_of_image_wrt_patch))
            print('bbox_of_patch_wrt_image: {}'.format(bbox_of_patch_wrt_image))
            raise MunglinkerDataError(e)

        bbox_of_f_wrt_patch = bbox_intersection(mungo_from.bounding_box, bbox_patch)
        if bbox_of_f_wrt_patch is None:
            raise MunglinkerDataError('Cannot generate patch for given FROM object {}/{}'
                                      ' -- no intersection with patch {}!'
                                      ''.format(mungo_from.uid, mungo_from.bounding_box,
                                                bbox_patch))
        bbox_of_patch_wrt_f = bbox_intersection(bbox_patch, mungo_from.bounding_box)

        f_mask_t, f_mask_l, f_mask_b, f_mask_r = bbox_of_f_wrt_patch
        f_mask = mungo_from.mask[f_mask_t:f_mask_b, f_mask_l:f_mask_r]

        f_patch_t, f_patch_l, f_patch_b, f_patch_r = bbox_of_patch_wrt_f
        output[1][f_patch_t:f_patch_b, f_patch_l:f_patch_r] = f_mask

        bbox_of_t_wrt_patch = bbox_intersection(mungo_to.bounding_box, bbox_patch)
        if bbox_of_t_wrt_patch is None:
            raise MunglinkerDataError('Cannot generate patch for given TO object {}/{}'
                                      ' -- no intersection with patch {}!'
                                      ''.format(mungo_to.uid, mungo_to.bounding_box,
                                                bbox_patch))
        bbox_of_patch_wrt_t = bbox_intersection(bbox_patch, mungo_to.bounding_box)

        t_mask_t, t_mask_l, t_mask_b, t_mask_r = bbox_of_t_wrt_patch
        t_mask = mungo_to.mask[t_mask_t:t_mask_b, t_mask_l:t_mask_r]

        t_patch_t, t_patch_l, t_patch_b, t_patch_r = bbox_of_patch_wrt_t
        try:
            output[2][t_patch_t:t_patch_b, t_patch_l:t_patch_r] = t_mask
        except ValueError:
            print('symbol_to: {}'.format(mungo_to))
            print('--------------- Absolute bboxes ----------')
            print('Patch bbox: {}'.format((t, l, b, r)))
            print('   to_bbox: {}'
                  ''.format(mungo_to.bounding_box))
            print('--------- w.r.t. "To" object ---------')
            print('to_mask_bbox: {}'
                  ''.format((t_mask_t, t_mask_l, t_mask_b, t_mask_r)))
            print('--------- w.r.t. Patch bbox -----------')
            print('to_patch_bbox: {}'
                  ''.format((t_patch_t, t_patch_l, t_patch_b, t_patch_r)))
            raise

        return output


    @staticmethod
    def get_closest_objects(cropobjects: List[CropObject], threshold=100):
        """For each pair of cropobjects, compute the closest distance between their
        bounding boxes.

        :returns: A dict of dicts, indexed by objid, then objid, then distance.
        """
        document = cropobjects[0].doc
        if document in distances_cache_per_file:
            return distances_cache_per_file[document]

        close_objects = {}
        for c in cropobjects:
            close_objects[c] = []

        for c in cropobjects:
            for d in cropobjects:
                distance = cropobject_distance(c, d)
                if distance < threshold:
                    close_objects[c].append(d)
                    close_objects[d].append(c)

        # Remove duplicates from lists
        for key, neighbors in close_objects.items():
            unique_neighbors = list(dict.fromkeys(neighbors))
            close_objects[key] = unique_neighbors

        distances_cache_per_file[document] = close_objects
        return close_objects

    @staticmethod
    def negative_example_pairs(cropobjects,
                               threshold=THRESHOLD_NEGATIVE_DISTANCE,
                               max_per_object=MAX_NEGATIVE_EXAMPLES_PER_OBJECT,
                               grammar=None):
        """Samples pairs of cropobjects that are *not* connected by an edge.

        :param cropobjects: A list of MuNG objects available for pair sampling.

        :param threshold: Maximum distance for a pair of MuNGos to be considered.

        :param max_per_object: At most this many negative samples will be output
            per object.

        :param grammar: If given, will only add negative pairs such that an edge
            between them would be permitted by the grammar.

        :return: A list of tuples of (from, to) MuNG objects that are *not* linked.
        """
        close_neighbors = PairwiseMungoDataPool.get_closest_objects(cropobjects, threshold)

        # Exclude linked ones
        negative_example_pairs_dict = {}
        for c in close_neighbors:
            negative_example_pairs_dict[c] = [d for d in close_neighbors[c] if d.objid not in c.outlinks]

            # Filter with grammar.
            if grammar is not None:
                negative_example_pairs_dict[c] = [d for d in negative_example_pairs_dict[c]
                                                  if grammar.validate_edge(c.clsname, d.clsname)]

        # Downsample,
        # -----------
        # but intelligently: there should be more weight on closer objects, as they should
        # be represented more (should they?) [NOT IMPLEMENTED].
        if (max_per_object is not None) and (max_per_object > 0):
            for c in close_neighbors:
                random.shuffle(negative_example_pairs_dict[c])
                negative_example_pairs_dict[c] = negative_example_pairs_dict[c][:max_per_object]

        negative_examples = []
        for c in negative_example_pairs_dict:
            negative_examples.extend([(c, d) for d in negative_example_pairs_dict[c]])

        return negative_examples

    @staticmethod
    def positive_example_pairs(cropobjects):
        _cdict = {c.objid: c for c in cropobjects}
        positive_example_pairs = []
        for c in cropobjects:
            for o in c.outlinks:
                positive_example_pairs.append((c, _cdict[o]))
        return positive_example_pairs

    @staticmethod
    def get_object_pairs(cropobjects,
                         max_object_distance=THRESHOLD_NEGATIVE_DISTANCE,
                         max_negative_samples=MAX_NEGATIVE_EXAMPLES_PER_OBJECT,
                         grammar=None):
        negative_pairs = PairwiseMungoDataPool.negative_example_pairs(cropobjects,
                                                threshold=max_object_distance,
                                                max_per_object=max_negative_samples,
                                                grammar=grammar)
        positive_pairs = PairwiseMungoDataPool.positive_example_pairs(cropobjects)
        logging.info('Object pair extraction: positive: {}, negative: {}'
                     ''.format(len(positive_pairs), len(negative_pairs)))
        return negative_pairs + positive_pairs


    @staticmethod
    def __compute_patch_center(m_from, m_to):
        """Computing the patch center for the given pair
        of objects. Gets returned as (row, column) coordinates
        with respect to the input image.

        Option 1: take their center of gravity.

        Option 2: if intersection, take center of intersection bbox.
            If not, take center of line connecting closest points.

        """
        intersection_bbox = m_from.bbox_intersection(m_to.bounding_box)
        if intersection_bbox is not None:
            it, il, ib, ir = intersection_bbox
            i_center_x = (it + ib) // 2
            i_center_y = (il + ir) // 2
            p_center_x, p_center_y = i_center_x + m_from.top, i_center_y + m_from.left
        else:
            # The "closest point" computation, which can actually be implemented
            # as taking the middle of the bounding box that is the intersection
            # of extending the objects' bounding boxes towards each other.

            # Vertical: take max of top, min of bottom, sort
            p_bounds_vertical = max(m_from.top, m_to.top), min(m_from.bottom, m_to.bottom)
            # i_bounds_top, i_bounds_bottom = min(i_bounds_vertical), max(i_bounds_vertical)

            # Horizontal: max of left, min of right, sort
            p_bounds_horizontal = max(m_from.left, m_to.left), min(m_from.right, m_to.right)
            # i_bounds_left, i_bounds_right = min(i_bounds_horizontal), max(i_bounds_horizontal)

            p_center_x = sum(p_bounds_vertical) // 2
            p_center_y = sum(p_bounds_horizontal) // 2

        return p_center_x, p_center_y


##############################################################################

# Techcnically these methods are the same, but there might in the future
# be different validation checks.

def load_split(split_file):
    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl)
    return split


def load_config(config_file: str):
    with open(config_file, 'rb') as hdl:
        config = yaml.load(hdl)
    return config


def __load_mung(filename: str, exclude_classes: List[str]):
    mungos = parse_cropobject_list(filename)
    mung = NotationGraph(mungos)
    objects_to_exclude = [m for m in mungos if m.clsname in exclude_classes]
    for m in objects_to_exclude:
        mung.remove_vertex(m.objid)
    return mung


def __load_image(filename: str) -> np.ndarray:
    image = np.array(Image.open(filename).convert('1')).astype('uint8')
    return image


def load_munglinker_data_lite(mung_root: str, images_root: str,
                              include_names: List[str] = None,
                              max_items: int = None,
                              exclude_classes=None,
                              masks_to_bounding_boxes=False):
    """Loads the MuNGs and corresponding images from the given folders.
    All *.xml files in ``mung_root`` are considered MuNG files, all *.png
    files in ``images_root`` are considered image files.

    Use this to get data for initializing the PairwiseMungoDataPool.

    :param mung_root: Directory containing MuNG XML files.

    :param images_root: Directory containing underlying image files (png).

    :param include_names: Only load files such that their basename is in
        this list. Useful for loading train/test/validate splits.

    :param max_items: Load at most this many files.

    :param exclude_classes: When loading the MuNG, exclude notation objects
        that are labeled as one of these classes. (Most useful for excluding
        staff objects.)

    :param masks_to_bounding_boxes: If set, will replace the masks of the
        loaded MuNGOs with everything in the corresponding bounding box
        of the image. This is to make the training data compatible with
        the runtime outputs of RCNN-based detectors, which only output
        the bounding box, not the mask.

    :returns: mungs, images  -- a tuple of lists.
    """
    if exclude_classes is None:
        exclude_classes = {}

    all_mung_files = glob(mung_root + "/**/*.xml", recursive=True)
    mung_files_in_this_split = [f for f in all_mung_files if os.path.splitext(os.path.basename(f))[0] in include_names]

    all_image_files = glob(images_root + "/**/*.png", recursive=True)
    image_files_in_this_split = [f for f in all_image_files if
                                 os.path.splitext(os.path.basename(f))[0] in include_names]

    mungs = []
    images = []
    for mung_file, image_file in tqdm(zip(mung_files_in_this_split, image_files_in_this_split),
                                      desc="Loading mung/image pairs from disk",
                                      total=len(mung_files_in_this_split)):
        mung = __load_mung(mung_file, exclude_classes)
        mungs.append(mung)

        image = __load_image(image_file)
        images.append(image)

        # This is for training on bounding boxes,
        # which needs to be done in order to then process
        # R-CNN detection outputs with Munglinker trained on ground truth
        if masks_to_bounding_boxes:
            for mungo in mung.cropobjects:
                t, l, b, r = mungo.bounding_box
                image_mask = image[t:b, l:r]
                mungo.set_mask(image_mask)

        if max_items is not None:
            if len(mungs) >= max_items:
                break

    return mungs, images


def load_munglinker_data(mung_root, images_root, split_file,
                         config_file=None,
                         load_training_data=True,
                         load_validation_data=True,
                         load_test_data=False,
                         exclude_classes=None):
    """Loads the train/validation/test data pools for the MuNGLinker
    experiments.

    :param mung_root: Directory containing MuNG XML files.

    :param images_root: Directory containing underlying image files (png).

    :param split_file: YAML file that defines which items are for training,
        validation, and test.

    :param config_file: YAML file defining further experiment properties.
        Not used so far.

    :param load_training_data: Whether or not to load the training data
    :param load_validation_data: Whether or not to load the validation data
    :param load_test_data: Whether or not to load the test data

    :param exclude_classes: When loading the MuNG, exclude notation objects
        that are labeled as one of these classes. (Most useful for excluding
        staff objects.)

    :return: ``dict(train=training_pool, valid=validation_pool, test=test_pool)``
    """
    split = load_split(split_file)
    train_on_bounding_boxes = False

    if config_file is not None:
        config = load_config(config_file)
        data_pool_dict = config2data_pool_dict(config)

        if 'TRAIN_ON_BOUNDING_BOXES' in config:
            train_on_bounding_boxes = config['TRAIN_ON_BOUNDING_BOXES']

        validation_data_pool_dict = copy.deepcopy(data_pool_dict)
        validation_data_pool_dict['resample_train_entities'] = False
        if 'VALIDATION_MAX_NEGATIVE_EXAMPLES_PER_OBJECT' in config:
            validation_data_pool_dict['max_negative_samples'] = \
                config['VALIDATION_MAX_NEGATIVE_EXAMPLES_PER_OBJECT']
    else:
        # Default configuration from variables set at the start of this module
        data_pool_dict = {
            'max_edge_length': THRESHOLD_NEGATIVE_DISTANCE,
            'max_negative_samples': MAX_NEGATIVE_EXAMPLES_PER_OBJECT,
            'resample_train_entities': RESAMPLE_EACH_EPOCH,
            'patch_size': (PATCH_HEIGHT, PATCH_WIDTH),
            'zoom': IMAGE_ZOOM
        }
        validation_data_pool_dict = copy.deepcopy(data_pool_dict)
        validation_data_pool_dict['resample_train_entities'] = False

    training_pool = None
    validation_pool = None
    test_pool = None

    if load_training_data:
        print("Loading training data...")
        tr_mungs, tr_images = load_munglinker_data_lite(mung_root, images_root,
                                                        include_names=split['train'],
                                                        exclude_classes=exclude_classes,
                                                        masks_to_bounding_boxes=train_on_bounding_boxes)
        training_pool = PairwiseMungoDataPool(mungs=tr_mungs, images=tr_images, **data_pool_dict)

    if load_validation_data:
        print("Loading validation data...")
        va_mungs, va_images = load_munglinker_data_lite(mung_root, images_root,
                                                        include_names=split['valid'],
                                                        exclude_classes=exclude_classes,
                                                        masks_to_bounding_boxes=train_on_bounding_boxes)
        validation_pool = PairwiseMungoDataPool(mungs=va_mungs, images=va_images, **validation_data_pool_dict)

    if load_test_data:
        print("Loading test data...")
        te_mungs, te_images = load_munglinker_data_lite(mung_root, images_root,
                                                        include_names=split['test'],
                                                        exclude_classes=exclude_classes,
                                                        masks_to_bounding_boxes=train_on_bounding_boxes)
        test_pool = PairwiseMungoDataPool(mungs=te_mungs, images=te_images, **data_pool_dict)

    return dict(train=training_pool, valid=validation_pool, test=test_pool)


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
    _start_time = time.time()

    # Your code goes here
    mung_root = '/Users/hajicj/data/MUSCIMA++/v1.0.1/data/cropobjects_complete'
    images_root = '/Users/hajicj/data/MUSCIMA++/v0.9/data/fulls'

    resources_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources')
    grammar_file = os.path.join(resources_root, 'mff-muscima-mlclasses-annot.deprules')
    grammar = load_grammar(grammar_file)

    mungs, images = load_munglinker_data_lite(mung_root, images_root,
                                              max_items=1,
                                              exclude_classes=_CONST.STAFFLINE_CROPOBJECT_CLSNAMES)

    data_pool = PairwiseMungoDataPool(mungs=mungs, images=images, grammar=grammar)
    print('Entities after loading data pool: {}'.format(len(data_pool.train_entities)))

    import matplotlib.pyplot as plt

    batch_size = 1
    n_batches = 10
    n_positive = 0
    for k in range(10):
        X, y = data_pool[k * batch_size:(k + 1) * batch_size]
        X0_sum = np.sum(X[0], axis=0)
        plt.imshow(X0_sum, cmap='gray', interpolation='nearest')
        plt.show()
        n_positive += y.sum()

    print('Positive example ratio: {}'.format(n_positive / (n_batches * batch_size)))

    _end_time = time.time()
    logging.info('data_pools.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
