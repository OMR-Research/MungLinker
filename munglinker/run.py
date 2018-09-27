#!/usr/bin/env python
"""This is a script that applies a trained e2e OMR model to an input image
or a directory of input images, and outputs the corresponding MIDI file(s).
"""
from __future__ import print_function, unicode_literals
import argparse
import collections
import logging
import os
import time

import numpy as np
import pickle
from scipy.misc import imread, imsave

from muscima.inference import play_midi

import torch
from torch.autograd import Variable

from munglinker.augmentation import ImageAugmentationProcessor
# from munglinker.model import FCN
# from munglinker.model import apply_on_image, apply_on_image_window
# from munglinker.model import apply_model
# from munglinker.model import ensure_shape_divisible, set_image_as_variable
# from munglinker.image_normalization import auto_invert, stretch_intensity
# from munglinker.image_normalization import ImageNormalizer
# from munglinker.utils import lasagne_fcn_2_pytorch_fcn
from munglinker.utils import generate_random_mm
from munglinker.utils import midi_matrix_to_midi

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################

class FakeModel(object):
    """Mock E2E OMR model. The outptut representation
    is a MIDI matrix: 128 rows, variable number of columns.

    The run() method returns a random MIDI matrix."""
    def run(self, image):
        n_cols = image.shape[1]
        random_mm = generate_random_mm((128, n_cols / 4), onset_density=100 / ((128 * n_cols) + 100))
        return random_mm

    def cuda(self):
        pass


class E2EOMRRunner(object):
    """The E2EOMRRunner defines the end-to-end OMR interface. It has a run()
    method that converts an image into MIDI, which is its main interface.
    """
    def __init__(self, model):
        """Initialize the E2E OMR runner.

        :param model: An instance that has a ``run(self, image)`` method
            that accepts a 2-D numpy array and outputs a (2-D) MIDI matrix.
        """
        self.model = model

    def run(self, staff_image):
        """Processes the image and outputs MIDI.

        :returns: A ``midiutil.MidiFile.MIDIFile`` object.
        """
        prep_image = self.prepare_image(staff_image)
        output_repr = self.model.run(prep_image)
        midi = self.model_output_to_midi(output_repr)
        return midi

    def prepare_image(self, staff_image):
        return staff_image

    def model_output_to_midi(self, output_repr):
        return midi_matrix_to_midi(output_repr)


##############################################################################


def show_result(img, midi_matrix):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap="gray")
    plt.ylabel(img.shape[0])
    plt.xlabel(img.shape[1])
    # plt.colorbar()

    # plt.subplot(3, 1, 2)
    # plt.imshow(spec[0][0], cmap="viridis", origin="lower", aspect="auto")
    # plt.ylabel(spec[0][0].shape[0])
    # plt.xlabel(spec[0][0].shape[1])
    # plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(midi_matrix, cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
    plt.ylabel(midi_matrix.shape[0])
    plt.xlabel(midi_matrix.shape[1])
    plt.colorbar()

    plt.show()


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--model', required=True,
                        help='The name of the model that you wish to use.')
    parser.add_argument('-p', '--params', required=True,
                        help='The state dict that should be loaded for this model.')

    parser.add_argument('-i', '--input_image',
                        help='A single-system input image for which MIDI should'
                             ' be output. This is the simplest input mode.')
    parser.add_argument('-o', '--output_midi',
                        help='The MIDI should be exported to this file.')
    parser.add_argument('--play', action='store_true',
                        help='If set, will play back the MIDI instead of saving'
                             ' to file.')

    parser.add_argument('--visualize', action='store_true',
                        help='If set, will plot the image and output MIDI.')

    parser.add_argument('--input_dir',
                        help='A directory with single-system input images. For'
                             ' each of these, a MIDI will be produced. Use'
                             ' instead of --input_image for batch processing.'
                             ' [NOT IMPLEMENTED]')
    parser.add_argument('--output_dir',
                        help='A directory where the output MIDI files will be'
                             ' stored. Use together with --input_dir.'
                             ' [NOT IMPLEMENTED]')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    ##########################################################################

    logging.info('Loading model: {} [NOT IMPLEMENTED]'.format(args.model))
    model = FakeModel()

    logging.info('Loading model params from state dict: {0} [NOT IMPLEMENTED]'.format(args.params))
    # params = torch.load(args.params)
    # model.load_state_dict(params)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logging.info('\tModel: CUDA available, moving to GPU')
        model.cuda()

    logging.info('Initializing runner...')
    runner = E2EOMRRunner(model=model)

    logging.info('Loading image: {}'.format(args.input_image))
    img = imread(args.input_image, mode='L')

    logging.info('Running OMR')
    mf = runner.run(img)

    if args.visualize:
        midi_matrix = model.run(img)
        show_result(img, midi_matrix)

    if args.play:
        logging.info('Playing output MIDI')
        play_midi(mf,
                  tmp_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'tmp'))

    else:
        logging.info('Saving output MIDI to {}'.format(args.output_midi))
        with open(args.output_midi, 'wb') as hdl:
            mf.writeFile(hdl)

    ##########################################################################

    _end_time = time.clock()
    logging.info('run.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
