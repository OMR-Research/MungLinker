"""This module implements data augmentation. Use the ImageAugmentationProcessor
class as a wrapper.

The transformations are parametrized by a single random value between -1 and 1,
and a *maximum extent* of the given transformation. The random value (``rval``)
is then interpreted as a fraction of this extent, in one or the other direction
when applicable (or its absolute value is taken when it does not make sense
to go in the "opposite direction", e.g. dilation). The random value is drawn
uniformly.

For example, if the ``rotation`` parameter is set to ``numpy.pi / 6``,
the maximum rotation will be up to 30 degrees, both left and right,
and the actual values will be distributed uniformly on the (-30, 30) interval.

If you do not want to draw the random values uniformly, you can of course
provide them manually to the ``process()`` method, inputting your
values into the ``rdict`` argument. See the attribute ``rval_dict_keys``
for the list of keys.

If run as a script, will generate a set of augmented images into an output dir.
Operates on a specified set of labels and image fnames. All label masks/images
for a given fname will be transformed identically, so that the generated
images can be simply used for evaluating and/or training the models (although
we do not recommend training with pre-defined augmentations; it's much better
to generate them on the fly -- see train.py).

A current limitation is that one image can only be transformed once,
otherwise it will suffer a naming conflict.
"""
from __future__ import print_function, unicode_literals

import argparse
import logging
import os
import pickle

import numpy.random
import scipy.misc
import time

import cv2

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class ImageAugmentationProcessor(object):
    """The ImageAugmentationProcessor class wraps the processing
    of an image when data augmentation is requested. Its initialization
    arguments are parameters of the augmentation process. The following
    input image transformations are available:

    * Vertical dilation
    * Horizontal dilation
    * Scaling
    * Rotation
    * Gaussian noise
    * Gaussian blur
    * Elastic deformation

    The transformations are applied *in this order*.
    """

    def __init__(self,
                 scaling_magnitude=0,
                 rotation=0,
                 curvature=0,
                 gaussian_noise=0,
                 gaussian_blur=0,
                 elastic_deformation=0,
                 dilation_vertical=0,
                 dilation_horizontal=0):
        """Initialize ImageAugmentationProcessor with the given parameters.
        If a transformation control parameter is set to ``0``, the given
        transformation will not be performed.

        :param curvature: The maximum angle of the wedge delimited by
            the bottom upwards curve, in radians.

            0 means the baseline is always straight,
            ``pi`` means that the line can be a perfect half-circle
            (which is useless, though). Optimal values will be around
            ``pi / 6``.

            In case the angle comes out negative, the curve is convex
            instead of concave.

        """
        self.scaling_magnitude = scaling_magnitude
        self.rotation = rotation
        self.curvature = curvature
        self.gaussian_noise = gaussian_noise
        self.gaussian_blur = gaussian_blur
        self.elastic_deformation = elastic_deformation
        self.dilation_vertical = dilation_vertical
        self.dilation_horizontal = dilation_horizontal

        self.rval_dict_keys = ['dilation_vertical',
                               'dilation_horizontal',
                               'scaling_magnitude',
                               'rotation', ]

    def generate_autoname(self):
        autoname = 'ImgAugP___scale={0}___rot={1}___dilh={2}___dilv={3}' \
                   ''.format(self.scaling_magnitude, self.rotation,
                             self.dilation_horizontal, self.dilation_vertical)
        return autoname

    def draw_rval_dict(self):
        rval_dict = {k: numpy.random.rand() * 2 - 1 for k in self.rval_dict_keys}
        return rval_dict

    def process(self, images, rval_dict=None, _debug_plot=False):
        """Applies all the transformations specified by the Augmenter's
        parameters.

        :param images: An iterable of images that should be transformed.

        :param rval_dict: If set, will apply the transformations with the given
            parameters instead of randomly choosing them from the (-1, 1) interval.
            All the random value dict keys should be from the interval (-1, 1).

        :returns: A list of transformed images.
        """
        if len(images) == 0:
            return images

        if _debug_plot:
            import matplotlib.pyplot as plt
            for img in images:
                plt.imshow(img, cmap='gray', interpolation='nearest')
                plt.show()

        if not rval_dict:
            rval_dict = self.draw_rval_dict()
            # ### DEBUG
            # print(rval_dict)

        for k in self.rval_dict_keys:
            if k not in rval_dict:
                rval_dict[k] = None

        if self.dilation_vertical:
            images = self.dilate_vertically(images,
                                            rval=rval_dict['dilation_vertical'])
            # print('DV: shapes: {0}'.format([img.shape for img in images]))

        if self.dilation_horizontal:
            images = self.dilate_horizontally(images,
                                              rval=rval_dict['dilation_horizontal'])
            # print('DH: shapes: {0}'.format([img.shape for img in images]))

        if self.curvature:
            images = self.curve(images)
            # print('Cr: shapes: {0}'.format([img.shape for img in images]))

        if self.scaling_magnitude:
            images = self.scale(images,
                                rval=rval_dict['scaling_magnitude'])
            # print('SC: shapes: {0}'.format([img.shape for img in images]))

        if self.rotation:
            images = self.rotate(images, rval=rval_dict['rotation'])
            # print('Rot: shapes: {0}'.format([img.shape for img in images]))

        if self.elastic_deformation:
            images = self.elastic_deform(images)
            # print('El: shapes: {0}'.format([img.shape for img in images]))

        if self.gaussian_noise:
            images = self.noise(images)
            # print('GN: shapes: {0}'.format([img.shape for img in images]))

        if self.gaussian_blur:
            images = self.blur(images)
            # print('GB: shapes: {0}'.format([img.shape for img in images]))

        if _debug_plot:
            import matplotlib.pyplot as plt
            for img in images:
                plt.imshow(img, cmap='gray', interpolation='nearest')
                plt.show()

        # print('Ouptut shapes: {0}'.format([img.shape for img in images]))

        return images

    def scale(self, images, rval=None):
        return self.scale_up(images, rval=rval)

    def scale_down(self, images, rval=None):
        # Scaling down requires knowing more than just the current window:
        # we need its surroundings.
        raise NotImplementedError()

    def scale_up(self, images, rval=None):
        # pick random scale
        if not rval:
            rval = numpy.random.random() # * 2 - 1

        s = (numpy.abs(rval) * self.scaling_magnitude) + 1
        rows, cols = images[0].shape
        t_rows, t_cols = int(rows * s), int(cols * s)
        d_rows, d_cols = (t_rows - rows) // 2, (t_cols - cols) // 2
        images = [cv2.resize(image.astype('uint8'), (int(cols * s), int(rows * s)),
                             fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
                  for image in images]
        # Centering back
        images_recropped = [image[d_rows:d_rows+rows, d_cols:d_cols+cols] for image in images]
        return images_recropped

    def rotate(self, images, rval=None):
        if not rval:
            rval = numpy.random.random() * 2 - 1
        s = (rval * 180) * (self.rotation / numpy.pi)
        rows, cols = images[0].shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), s, 1)
        images = [cv2.warpAffine(image, M, dsize=(cols, rows)) for image in images]
        # images = [scipy.misc.imresize(image, s, interp='nearest')
        #           for image in images]
        return images

    def noise(self, images):
        return images

    def blur(self, images):
        return images

    def elastic_deform(self, images):
        return images

    def dilate_vertically(self, images, rval=None):
        if not rval:
            rval = numpy.random.random() * 2 - 1
        # Derive randomized size of structuring element from a number in (-1, 1)
        rsize = int((rval + 1) / 2 * (self.dilation_vertical + 1))
        if rsize == 0:
            return images

        selem_edge = 2 * rsize + 1
        selem = numpy.zeros(shape=(selem_edge, selem_edge), dtype='uint8')
        selem[:, rsize] = 1

        images = [cv2.dilate(image, kernel=selem)
                  for image in images]
        return images

    def dilate_horizontally(self, images, rval=None):
        if not rval:
            rval = numpy.random.random() * 2 - 1
        # Derive randomized size of structuring element from a number in (-1, 1)
        rsize = int((rval + 1) / 2 * (self.dilation_vertical + 1))
        if rsize == 0:
            return images

        selem_edge = 2 * rsize + 1
        selem = numpy.zeros(shape=(selem_edge, selem_edge), dtype='uint8')
        selem[rsize, :] = 1

        images = [cv2.dilate(image, kernel=selem)
                  for image in images]
        return images

    def curve(self, images):
        return images

        # Assume the images all have the same size.
        h, w = images[0].shape

        _h_mesh_step = h / 100
        _h_mesh_points = numpy.array(list(range(0, h, _h_mesh_step)) + [h - 1])

        _w_mesh_step = w / 10
        _w_mesh_points = numpy.array(list(range(0, w, _w_mesh_step)) + [w - 1])

        # Compute the arc radius.

        # Curvature is in radians, so this comes out at a maximum of (-pi, pi)
        # if self.curvature == numpy.pi
        # Numpy sin() and cos() all work on radians.
        alpha_rad = (numpy.random.rand() * 2 - 1) * self.curvature
        _is_angle_negative = False
        if alpha_rad < 0:
            _is_angle_negative = True
            alpha_rad *= -1

        arc_radius = (w / 2.0) * numpy.sqrt(1 + numpy.cos(alpha_rad / 2) ** 2)

        # Generate the baseline.

        # For each column c, the corresponding baseline point lies
        # on a circle. This circle has an imaginary center S so that
        # it forms a triangle with the bottom left and the bottom right
        # corners, such that the distances of both corners to S are
        # the same.
        #
        # The formula to calculate the vertical displacement d_h for
        # a given column c is:
        #
        #  $ d_h(c) = y_c - \sqrt(a^2 - (w / 2)^2) $
        #
        # Where $a$ is the arc radius, $w$ is the width of the picture,
        # and y_c is the vertical coordinate of the given baseline
        # curve point relative to S. The formula for $y_c$ is:
        #
        #  $ y_c = \sqrt(a^2 - (c - w/2)^2) $
        #
        # which follows from the formula for a circle with radius $a%
        # from analytical geometry:
        #
        #  $ (x / a)^2 + (y / a)^2 = 1 $

        a = arc_radius
        dh_center = numpy.sqrt(a ** 2 - (w / 2) ** 2)
        _w_mesh_dh = numpy.zeros((len(_w_mesh_points)))
        for i, c in enumerate(_w_mesh_points):
            y_c = numpy.sqrt(a ** 2 - (c - w / 2) ** 2)
            dh_t = y_c - dh_center
            _w_mesh_dh[i] = dh_t


##############################################################################

def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-l', '--labels', action='store', nargs='+', required=True,
                        help='List of labels to which to apply the transformations.')
    parser.add_argument('-f', '--fnames', action='store', nargs='+', required=True,
                        help='If set, will read a file listing image names'
                             ' which should be excluded from training'
                             ' (one name per line). The names should be'
                             ' given w.r.t. to the label directory.'
                             ' Useful for maintaining consistent test'
                             ' sets.')
    parser.add_argument('-i', '--input_root', action='store', required=True,
                        help='The root directory which contains the masks'
                             ' for individual labels.')
    parser.add_argument('-o', '--output_dir', action='store', required=True,
                        help='The directory into which the transformed images'
                             ' should be saved. This directory will contain'
                             ' a subdirectory for each of the --labels that'
                             ' were processed. Name this directory well, to'
                             ' reflect what transformations are applied.')
    parser.add_argument('--output_dir_autoname', action='store_true',
                        help='If set, will (a) interpret --otuput_dir as the'
                             ' *parent* directory of where the outputs should go,'
                             ' and (b) generate the actual output directory name'
                             ' from the augmentation parameters.')
    parser.add_argument('--export_rvals', action='store',
                        help='If set, will pickle a dict of the random value dicts'
                             ' used to transform the masks per fname. The keys'
                             ' will be fnames, the values will be the rval dicts.')

    parser.add_argument('-s', '--scaling_magnitude', type=float, default=0.4,
                        help='The maximum extent of scaling, both up and down.')
    parser.add_argument('--rotation', type=float, default=numpy.pi / 6.0,
                        help='The maximum extent of rotation, in radians.')
    parser.add_argument('--dilation_vertical', type=int, default=5,
                        help='Maximum extent of vertical dilation.')
    parser.add_argument('--dilation_horizontal', type=int, default=2,
                        help='Maximum extent of horizontal dilation.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    logging.info('Initialize augmenter...')
    augmenter = ImageAugmentationProcessor(
        dilation_vertical=args.dilation_vertical,
        dilation_horizontal=args.dilation_horizontal,
        scaling_magnitude=args.scaling_magnitude,
        rotation=args.rotation,
    )

    if args.output_dir_autoname:
        autoname = augmenter.generate_autoname()
        logging.info('Generated autoname: {0}'.format(autoname))
        args.output_dir = os.path.join(args.output_dir, autoname)

    logging.info('Initialize output directories, with output dir: {0}'
                 ''.format(args.output_dir))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    for label in args.labels:
        lpath = os.path.join(args.output_dir, label)
        if not os.path.isdir(lpath):
            os.makedirs(lpath)

    # Initialize random value dict persistence.
    rval_dicts = {}

    # For each fname:
    for fname in args.fnames:
        logging.info('Processing fname: {0}'.format(fname))

        #  Generate rval dict.
        rdict = augmenter.draw_rval_dict()
        rval_dicts[fname] = rdict

        #  For each label:
        for label in args.labels:
            #   Load image indexed by fname and label.
            img_fname = fname + '.png'
            img_path = os.path.join(args.input_root, label, img_fname)
            if not os.path.isfile(img_path):
                raise OSError('Image file for fname {0}, label {1} does not exists: {2}'
                              ''.format(fname, label, img_path))
            image = scipy.misc.imread(img_path, mode='L')

            #   Process image using given rval dict.
            transformed_image = augmenter.process([image], rval_dict=rdict)[0]

            #   Save processed image to output directory.
            output_img_fname = fname + '.png'
            output_path = os.path.join(args.output_dir, label, output_img_fname)
            scipy.misc.imsave(output_path, transformed_image)

    if args.export_rvals:
        logging.info('Exporting rvals dict to {0}'.format(args.export_rvals))
        with open(args.export_rvals, 'w') as hdl:
            pickle.dump(rval_dicts, hdl, protocol=pickle.HIGHEST_PROTOCOL)

    _end_time = time.clock()
    logging.info('augmentation.py done in {0:.3f} s'.format(_end_time - _start_time))


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)

