#!/usr/bin/env python
"""This is a script that applies a trained e2e OMR model to an input image
or a directory of input images, and outputs the corresponding MIDI file(s).
"""
from __future__ import print_function, unicode_literals
import argparse
import collections
import copy
import logging
import os
import time

import numpy as np
import pickle

from muscima.graph import NotationGraph
from muscima.io import parse_cropobject_list, export_cropobject_list
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
from munglinker.data_pool import PairwiseMungoDataPool, load_config
from munglinker.model import PyTorchNetwork
from munglinker.mung2midi import build_midi
from munglinker.utils import generate_random_mm, select_model, config2data_pool_dict, MockNetwork
from munglinker.utils import midi_matrix_to_midi

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


class MunglinkerRunner(object):
    """The MunglinkerRunner defines the Munglinker component interface. It has a run()
    method that takes a MuNG (the whole graph) and outputs a new MuNG with the same
    objects, but different edges.
    """
    def __init__(self, model, config, runtime_batch_iterator,
                 replace_all_edges=True):
        """Initialize the Munglinker runner.

        :param model: A PyTorchNetwork() object with a net. Its predict()
            method is called, with a data pool that is constructed on the
            fly from the provided images & mungs, and with the batch iterator
            provided to this  __init__() method.

        :param config: The configuration that was used to train the given model.
            Contains important things like patch size.

        :param runtime_batch_iterator: The prepared

        """
        self.model = model
        self.config = config
        self.runtime_batch_iterator = runtime_batch_iterator

        # We pre-build the parameters that are used to wrap the input data
        # into a data pool.
        data_pool_dict = config2data_pool_dict(self.config)
        data_pool_dict['max_negative_samples'] = -1
        data_pool_dict['resample_train_entities'] = False
        if 'grammar' not in data_pool_dict:
            logging.warning('MunglinkerRunner expects a grammar to restrict'
                            ' edge candidates. Without a grammar, it will take'
                            ' a long time, since all possible object pairs'
                            ' will be tried. (This is fine if you trained without'
                            ' the grammar restriction, obviously.)')
        self.data_pool_dict = data_pool_dict

        self.replace_all_edges = replace_all_edges

    def run(self, image, mung):
        """Processes the image and outputs MIDI.

        :returns: A ``midiutil.MidiFile.MIDIFile`` object.
        """
        data_pool = self.build_data_pool(image, mung)
        mungo_pairs, output_classes = self.model.predict(data_pool,
                                                         self.runtime_batch_iterator)
        logging.info('Prediction: {} positive'.format(output_classes.sum()))

        # Since the runner only takes one image & MuNG at a time,
        # we have the luxury that all the mung pairs belong to the same
        # document, and we can just re-do the edges.
        mungo_copies = [copy.deepcopy(m) for m in mung.cropobjects]
        if self.replace_all_edges:
            for m in mungo_copies:
                m.outlinks = []
                m.inlinks = []

        new_mung = NotationGraph(mungo_copies)
        for mungo_pair, has_edge in zip(mungo_pairs, output_classes):
            if has_edge:
                logging.debug('Adding edge: {} --> {}'.format(mungo_pair[0].objid,
                                                              mungo_pair[1].objid))
                mungo_fr, mungo_to = mungo_pair
                new_mung.add_edge(mungo_fr.objid, mungo_to.objid)
            else:
                mungo_fr, mungo_to = mungo_pair
                if new_mung.has_edge(mungo_fr.objid, mungo_to.objid):
                    new_mung.remove_edge(mungo_fr.objid, mungo_to.objid)

        return new_mung

    def build_data_pool(self, image, mung):
        data_pool = PairwiseMungoDataPool(mungs=[mung], images=[image],
                                          **self.data_pool_dict)
        return data_pool

    def model_output_to_midi(self, output_repr):
        return midi_matrix_to_midi(output_repr)


##############################################################################


def show_result(*args, **kwargs):
    raise NotImplementedError()

##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--model', required=True,
                        help='The name of the model that you wish to use.')
    parser.add_argument('-p', '--params', required=True,
                        help='The state dict that should be loaded for this model.'
                             ' Note that you have to make sure you are loading'
                             ' a state dict for the right model architecture.')
    parser.add_argument('-c', '--config', required=True,
                        help='The config file that controls how inputs'
                             ' to the network will be extracted from MuNGOs.')

    parser.add_argument('-i', '--input_image', required=True,
                        help='A single-system input image for which MIDI should'
                             ' be output. This is the simplest input mode.'
                             ' If a directory is given, it will run over all'
                             ' images in that directory, expecting --input_mung'
                             ' to also be a directory with correspondingly named'
                             ' MuNG files (like for training).')
    parser.add_argument('-g', '--input_mung', required=True,
                        help='A MuNG XML file. The edges inoinks/outlinks in'
                             ' the file are ignored, unless the --retain_edges'
                             ' flag is set [NOT IMPLEMENTED]. If this is a'
                             ' directory, it will run over all MuNGs in that'
                             ' directory, expecting --input_image to also'
                             ' be a directory with correspondingly named'
                             ' image files (like for training).')

    parser.add_argument('-o', '--output_mung', required=True,
                        help='The MuNG with inferred edges should be exported'
                             ' to this file. If this is a directory, will instead'
                             ' export all the output MuNGs here, with names copied'
                             ' from the input MuNGs.')

    parser.add_argument('--visualize', action='store_true',
                        help='If set, will plot the image and output MIDI'
                             '[NOT IMPLEMENTED].')
    parser.add_argument('--batch_size', type=int, action='store', default=10,
                        help='The runtime iterator batch size.')

    parser.add_argument('--mock', action='store_true',
                        help='If set, will not load a real model and just run'
                             ' a mock prediction using MockNetwork.predict()')

    parser.add_argument('--play', action='store_true',
                        help='If set, will run MIDI inference over the output'
                             ' MuNG and play the result. [NOT IMPLEMENTED]')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.time()

    ##########################################################################
    # First we prepare the model

    logging.info('Loading config: {}'.format(args.config))
    config = load_config(args.config)

    logging.info('Loading model: {}'.format(args.model))
    model_mod = select_model(args.model)
    build_model_fn = model_mod.get_build_model()
    net = build_model_fn()

    runtime_batch_iterator = model_mod.runtime_batch_iterator(batch_size=args.batch_size)

    if args.mock:
        logging.info('Using mock network, so no parameters will be loaded.')
        model = MockNetwork()
    else:
        logging.info('Loading model checkpoint from state dict: {0}'.format(args.params))
        checkpoint = torch.load(args.params)
        net.load_state_dict(checkpoint['model_state_dict'])
        model = PyTorchNetwork(net=net)

    ########################################################
    # Prepare runner

    logging.info('Initializing runner...')
    runner = MunglinkerRunner(model=model,
                              config=config,
                              runtime_batch_iterator=runtime_batch_iterator,
                              replace_all_edges=True)

    ########################################################
    # Load data & run
    image_files = []
    input_mung_files = []

    if os.path.isfile(args.input_image):
        logging.info('Loading image: {}'.format(args.input_image))
        image_files.append(args.input_image)
    elif os.path.isdir(args.input_image):
        raise NotImplementedError
    else:
        raise OSError('Input image(s) not found: {}'.format(args.input_image))

    if os.path.isfile(args.input_mung):
        logging.info('Loading MuNG: {}'.format(args.input_mung))
        input_mung_files.append(args.input_mung)
    elif os.path.isdir(args.input_mung):
        raise NotImplementedError
    else:
        raise OSError('Input MuNG(s) not found: {}'.format(args.input_mung))

    ########################################################
    # Run munglinker model

    output_mungs = []
    for i, (image_file, input_mung_file) in enumerate(zip(image_files, input_mung_files)):

        img = imread(image_file, mode='L')

        input_mungos = parse_cropobject_list(input_mung_file)
        input_mung = NotationGraph(input_mungos)

        logging.info('Running Munglinker: {} / {}'.format(i, len(image_files)))
        output_mung = runner.run(img, input_mung)
        output_mungs.append(output_mung)

    ##########################################################################
    # And deal with the output:

    if args.visualize:
        logging.info('Visualization not implemented!!!')
        pass

    if args.play:
        logging.info('Playback not implemented!!!')
        # mf = build_midi(cropobjects=output_mung.cropobjects)
        # with open(output_path, 'wb') as stream_out:
        #     mf.writeFile(stream_out)
        pass

    ##########################################################################
    # Save output (TODO: refactor this into the processing loop)

    if os.path.isdir(args.output_mung):
        output_mung_files = [os.path.join(args.output_mung, os.path.basename(f))
                             for f in input_mung_files]
    else:
        output_mung_files = [args.output_mung]

    for output_mung_file, output_mung in zip(output_mung_files, output_mungs):
        logging.info('Saving output MuNG to: {}'.format(output_mung_file))
        with open(output_mung_file, 'w') as hdl:
            hdl.write(export_cropobject_list(output_mung.cropobjects))

    # (No evaluation in this script.)

    _end_time = time.time()
    logging.info('run.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
