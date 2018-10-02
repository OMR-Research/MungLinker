#!/usr/bin/env python
"""This is a script that..."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import os
import random
import time

import numpy as np

from muscima.cropobject import cropobject_distance, bbox_intersection
from muscima.io import parse_cropobject_list
from muscima.graph import NotationGraph
from muscima.inference_engine_constants import _CONST

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################

# Data point sampling parameters
THRESHOLD_NEGATIVE_DISTANCE = 200
MAX_NEGATIVE_EXAMPLES_PER_OBJECT = None

# The size of the image patches that are ouptut
PATCH_HEIGHT = 256
PATCH_WIDTH = 512

# The rescaling factor that is applied before the patch is extracted
# (In effect, setting this to 0.5 downscales by a factor of 2, so
#  the effective window w.r.t. the input image will be twice the specified
#  PATCH_HEIGHT, PATCH_WIDTH.)
IMAGE_ZOOM = 0.5

# Randomly moves the patch this many pixels away from the midpoint
# between the two sampled objects.
MAX_PATCH_DISPLACEMENT = 0


##############################################################################
# These functions are concerned with data point extraction for parser training.
# Actually, it could be wrapped by a DataPool.

def symbol_distances(cropobjects):
    """For each pair of cropobjects, compute the closest distance between their
    bounding boxes.

    :returns: A dict of dicts, indexed by objid, then objid, then distance.
    """
    _start_time = time.clock()
    distances = {}
    for c in cropobjects:
        distances[c] = {}
        for d in cropobjects:

            if d not in distances:
                distances[d] = {}
            if d not in distances[c]:
                delta = cropobject_distance(c, d)
                distances[c][d] = delta
                distances[d][c] = delta
    print('Distances for {0} cropobjects took {1:.3f} seconds'
          ''.format(len(cropobjects), time.clock() - _start_time))
    return distances


def get_close_objects(dists, threshold=100):
    """Returns a dict: for each cropobject a list of cropobjects
    that are within the threshold."""
    output = {}
    for c in dists:
        output[c] = []
        for d in dists[c]:
            if dists[c][d] < threshold:
                output[c].append(d)
    return output


def negative_example_pairs(cropobjects,
                           threshold=THRESHOLD_NEGATIVE_DISTANCE,
                           max_per_object=MAX_NEGATIVE_EXAMPLES_PER_OBJECT):
    dists = symbol_distances(cropobjects)
    close_neighbors = get_close_objects(dists, threshold=threshold)
    # Exclude linked ones
    negative_example_pairs_dict = {}
    for c in close_neighbors:
        negative_example_pairs_dict[c] = [d for d in close_neighbors[c] if d.objid not in c.outlinks]

        # Downsample,
    # but intelligently: there should be more weight on closer objects, as they should
    # be represented more.
    if max_per_object is not None:
        for c in close_neighbors:
            random.shuffle(negative_example_pairs_dict[c])
            negative_example_pairs_dict[c] = negative_example_pairs_dict[c][:max_per_object]
    negative_example_pairs = []
    for c in negative_example_pairs_dict:
        negative_example_pairs.extend([(c, d) for d in negative_example_pairs_dict[c]])
    return negative_example_pairs


def positive_example_pairs(cropobjects):
    _cdict = {c.objid: c for c in cropobjects}
    positive_example_pairs = []
    for c in cropobjects:
        for o in c.outlinks:
            positive_example_pairs.append((c, _cdict[o]))
    return positive_example_pairs


def get_object_pairs(cropobjects,
                     max_object_distance=THRESHOLD_NEGATIVE_DISTANCE,
                     max_negative_samples=MAX_NEGATIVE_EXAMPLES_PER_OBJECT):

    negs = negative_example_pairs(cropobjects,
                                  threshold=max_object_distance,
                                  max_per_object=max_negative_samples)
    poss = positive_example_pairs(cropobjects)
    return negs + poss


##############################################################################


class PairwiseMungoDataPool:
    """This class implements the basic data pool for munglinker experiments
    that outputs just pairs of MuNG nodes from the same document. Using this
    pool means that your preparation function will have to deal with everything
    else, like having at its disposal also the appropriate image from which
    to get the input patch, if need be in your model.

    It is entirely sufficient for training the baseline decision trees without
    complications, though.
    """
    def __init__(self, mungs, images,
                 max_edge_length=THRESHOLD_NEGATIVE_DISTANCE,
                 max_negative_samples=MAX_NEGATIVE_EXAMPLES_PER_OBJECT,
                 patch_size=(PATCH_HEIGHT, PATCH_WIDTH),
                 zoom=IMAGE_ZOOM,
                 max_patch_displacement=MAX_PATCH_DISPLACEMENT):
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

        :param patch_size: What the size of the extracted patch should
            be (after applying zoom), specified as ``(rows, columns)``.

        :param zoom: The rescaling factor. Setting this to 0.5 means
            the image will be downscaled to half the height & width
            before the patch is extracted.

        :param max_patch_displacement: The patch center will be uniformly
            sampled from up to this many pixels (both vertically and
            horizontally) away from the true center point between
            the mungos pair.
        """
        self.mungs = mungs
        self.images = images

        self.max_edge_length = max_edge_length
        self.max_negative_samples = max_negative_samples

        self.prepare_train_entities()

        self.patch_size = patch_size
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]

    def __getitem__(self, key):
        # get batch
        if key.__class__ == int:
            key = slice(key, key + 1)

        batch_entities = self.train_entities[key]

        # Return the patches
        patches_batch = np.zeros((len(batch_entities), 3,
                                  self.patch_height, self.patch_width))
        targets = np.zeros(len(batch_entities))
        for i_entity, (i_image, i_mungo_pair) in enumerate(batch_entities):

            m_from, m_to = self._mungo_pair_map[i_mungo_pair]

            patch = self.prepare_train_patch(i_image, m_from, m_to)
            patches_batch[i_entity] = patch

            if m_to.objid in m_from.outlinks:
                targets[i_entity] = 1

        return [patches_batch, targets]

    def reset_batch_generator(self):
        """Reset data pool with new random reordering of ``train_entities``.
        """
        indices = np.random.permutation(len(self.train_entities))
        self.train_entities = self.train_entities[indices]

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
        for i_doc, mung in enumerate(self.mungs):
            object_pairs = get_object_pairs(
                mung.cropobjects,
                max_object_distance=self.max_edge_length,
                max_negative_samples=self.max_negative_samples)
            for (m_from, m_to) in object_pairs:
                self._mungo_pair_map.append((m_from, m_to))
                self.train_entities.append([i_doc, n_entities])
                n_entities += 1

    def prepare_train_patch(self, i_image, m_from, m_to):
        image = self.images[i_image]
        patch = self.get_X_patch(image, m_from, m_to)
        return patch

    def get_X_patch(self, image, mungo_from, mungo_to):
        m_vert, m_horz = self._compute_patch_center(mungo_from, mungo_to)
        patch_radius_v = self.patch_height // 2
        patch_radius_h = self.patch_width // 2
        t, l, b, r = m_vert - patch_radius_v, \
                     m_horz - patch_radius_h, \
                     m_vert + patch_radius_v, \
                     m_horz + patch_radius_h
        bbox_patch = t, l, b, r

        output = np.zeros((3, (b - t), (r - l)))
        try:
            output[0] = image[t:b, l:r] * 1
        except ValueError:
            print('Patch bounding box: {}'.format(bbox_patch))
            raise
        if output[0].max() > 1.0:
            output[0] = output[0] / output[0].max()

        bbox_of_f_wrt_patch = bbox_intersection(mungo_from.bounding_box, bbox_patch)
        if bbox_of_f_wrt_patch is None:
            raise ValueError('Cannot generate patch for given FROM object {}/{}'
                             ' -- no intersection with patch {}!'
                             ''.format(mungo_from.uid, mungo_from.bounding_box,
                                       bbox_patch))
        bbox_of_patch_wrt_f = bbox_intersection(bbox_patch, mungo_from.bounding_box)

        f_mask_t, f_mask_l, f_mask_b, f_mask_r = bbox_of_f_wrt_patch
        f_mask = mungo_from.mask[f_mask_t:f_mask_b, f_mask_l:f_mask_r]

        f_patch_t, f_patch_l, f_patch_b, f_patch_r = bbox_of_patch_wrt_f
        output[1][f_patch_t:f_patch_b, f_patch_l:f_patch_r] = f_mask
        if output[1].max() > 1.0:
            output[1] = output[1] / output[1].max()

        bbox_of_t_wrt_patch = bbox_intersection(mungo_to.bounding_box, bbox_patch)
        if bbox_of_t_wrt_patch is None:
            raise ValueError('Cannot generate patch for given TO object {}/{}'
                             ' -- no intersection with patch {}!'
                             ''.format(mungo_to.uid, mungo_to.bounding_box,
                                       bbox_patch))
        bbox_of_patch_wrt_t = bbox_intersection(bbox_patch, mungo_to.bounding_box)

        t_mask_t, t_mask_l, t_mask_b, t_mask_r = bbox_of_t_wrt_patch
        t_mask = mungo_to.mask[t_mask_t:t_mask_b, t_mask_l:t_mask_r]

        t_patch_t, t_patch_l, t_patch_b, t_patch_r = bbox_of_patch_wrt_t
        try:
            output[2][t_patch_t:t_patch_b, t_patch_l:t_patch_r] = t_mask
            if output[2].max() > 1.0:
                output[2] = output[2] / output[2].max()
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
    def _compute_patch_center(m_from, m_to):
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


def load_munglinker_data(mung_root, images_root,
                         max_items=None, exclude_classes=None):
    """Loads the MuNGs and corresponding images from the given folders.
    All *.xml files in ``mung_root`` are considered MuNG files, all *.png
    files in ``images_root`` are considered image files.

    Use this to get data for initializing the PairwiseMungoDataPool.

    :returns: mungs, images  -- a tuple of lists.
    """
    if exclude_classes is None:
        exclude_classes = {}

    def _load_mung(filename, exclude_classes=exclude_classes):
        mungos = parse_cropobject_list(filename)
        mung = NotationGraph(mungos)
        objects_to_exclude = [m for m in mungos if m.clsname in exclude_classes]
        for m in objects_to_exclude:
            mung.remove_vertex(m.objid)
        return mung

    def _load_image(filename):
        import PIL.Image
        image = np.array(PIL.Image.open(filename).convert('L')).astype('uint8')
        return image

    mung_files = [os.path.join(mung_root, f) for f in os.listdir(mung_root)
                  if f.endswith('xml')]
    mung_ids = [os.path.splitext(os.path.basename(f))[0] for f in mung_files]

    image_files = [os.path.join(images_root, f) for f in os.listdir(images_root)
                   if f.endswith('png')]
    image_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

    mung_idxs = [idx for idx, item_id in enumerate(mung_ids) if item_id in image_ids]
    image_idxs = [idx for idx, item_id in enumerate(image_ids) if item_id in mung_ids]

    mungs = []
    images = []
    for mung_idx, image_idx in zip(mung_idxs, image_idxs):
        logging.info('Loading mung/image pair {}'.format((mung_idx, image_idx)))
        mung_file = mung_files[mung_idx]
        mung = _load_mung(mung_file)
        mungs.append(mung)

        image_file = image_files[image_idx]
        image = _load_image(image_file)
        images.append(image)

        if max_items is not None:
            if len(mungs) >= max_items:
                break

    return mungs, images

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
    mung_root = '/Users/hajicj/data/MUSCIMA++/v1.0.1/data/cropobjects_complete'
    images_root = '/Users/hajicj/data/MUSCIMA++/v0.9/data/fulls'

    mungs, images = load_munglinker_data(mung_root, images_root,
                                         max_items=1,
                                         exclude_classes=_CONST.STAFFLINE_CROPOBJECT_CLSNAMES)

    data_pool = PairwiseMungoDataPool(mungs=mungs, images=images)

    X, y = data_pool[0:4]

    import matplotlib.pyplot as plt
    X0 = X[0]
    X0_sum = np.sum(X[0], axis=0)
    print('X0_sum shape: {}'.format(X0_sum.shape))
    print('X0 sums: {}'.format(X0.sum(axis=1).sum(axis=1)))
    plt.imshow(X0_sum, cmap='gray', interpolation='nearest')
    plt.show()

    _end_time = time.clock()
    logging.info('data_pools.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
