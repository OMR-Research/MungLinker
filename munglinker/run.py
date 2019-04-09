"""This is a script that applies a trained e2e OMR model to an input image
or a directory of input images, and outputs the corresponding MIDI file(s).
"""
import argparse
import copy
import logging
import os
import time
from glob import glob
from typing import Dict

import torch
from PIL import Image
from muscima.cropobject import CropObject
from muscima.graph import NotationGraph, NotationGraphError
from muscima.io import parse_cropobject_list, export_cropobject_list

from munglinker.batch_iterators import PoolIterator
from munglinker.data_pool import PairwiseMungoDataPool, load_config
from munglinker.evaluate_notation_assembly_from_mung import evaluate_result
from munglinker.model import PyTorchNetwork
from munglinker.mung2midi import build_midi
from munglinker.utils import midi_matrix_to_midi
from munglinker.utils import select_model, config2data_pool_dict, MockNetwork
import numpy as np


class MunglinkerRunner(object):
    """The MunglinkerRunner defines the Munglinker component interface. It has a run()
    method that takes a MuNG (the whole graph) and outputs a new MuNG with the same
    objects, but different edges.
    """

    def __init__(self, model: PyTorchNetwork, config,
                 runtime_batch_iterator: PoolIterator):
        """Initialize the Munglinker runner.

        :param model: A PyTorchNetwork() object with a net. Its predict()
            method is called, with a data pool that is constructed on the
            fly from the provided images & mungs, and with the batch iterator
            provided to this  __init__() method.

        :param config: The configuration that was used to train the given model.
            Contains important things like patch size.

        :param runtime_batch_iterator:

        """
        self.model = model
        self.config = config
        self.runtime_batch_iterator = runtime_batch_iterator

        # We pre-build the parameters that are used to wrap the input data
        # into a data pool.
        data_pool_dict = config2data_pool_dict(self.config)
        data_pool_dict['max_negative_samples'] = -1
        if 'grammar' not in data_pool_dict:
            logging.warning('MunglinkerRunner expects a grammar to restrict'
                            ' edge candidates. Without a grammar, it will take'
                            ' a long time, since all possible object pairs'
                            ' will be tried. (This is fine if you trained without'
                            ' the grammar restriction, obviously.)')
        if 'TRAIN_ON_BOUNDING_BOXES' in self.config:
            self.masks_to_bounding_boxes = self.config['TRAIN_ON_BOUNDING_BOXES']

        self.data_pool_dict = data_pool_dict

    def run(self, image_file, mung: NotationGraph) -> NotationGraph:
        image = np.array(Image.open(image_file).convert('1')).astype('uint8')

        # This is for training on bounding boxes,
        # which needs to be done in order to then process
        # R-CNN detection outputs with Munglinker trained on ground truth
        if self.masks_to_bounding_boxes:
            for mungo in mung.cropobjects:
                t, l, b, r = mungo.bounding_box
                image_mask = image[t:b, l:r]
                mungo.set_mask(image_mask)

        data_pool = PairwiseMungoDataPool(mungs=[mung], images=[image], **self.data_pool_dict)
        mungos_from, mungos_to, output_classes = self.model.predict(data_pool, self.runtime_batch_iterator)

        # Since the runner only takes one image & MuNG at a time,
        # we have the luxury that all the mung pairs belong to the same
        # document, and we can just re-do the edges.
        mungo_copies = [copy.deepcopy(m) for m in mung.cropobjects]
        for m in mungo_copies:
            m.outlinks = []
            m.inlinks = []

        notation_graph = NotationGraph(mungo_copies)
        id_to_crop_object_mapping = {c.objid: c for c in notation_graph.cropobjects}
        for mungo_from, mungo_to, output_class in zip(mungos_from, mungos_to, output_classes):
            has_edge = output_class == 1
            if has_edge:
                self.add_edge_in_graph(mungo_from.objid, mungo_to.objid, id_to_crop_object_mapping)

        return notation_graph

    @staticmethod
    def add_edge_in_graph(from_node: CropObject, to_node: CropObject,
                          id_to_crop_object_mapping: Dict[int, CropObject]):
        """Add an edge between the MuNGOs with objids ``fr --> to``.
            If the edge is already in the graph, warns and does nothing."""
        if from_node not in id_to_crop_object_mapping:
            raise NotationGraphError('Cannot remove edge from node_id {0}: not in graph!'.format(from_node))
        if to_node not in id_to_crop_object_mapping:
            raise NotationGraphError('Cannot remove edge to node_id {0}: not in graph!'.format(to_node))

        if to_node in id_to_crop_object_mapping[from_node].outlinks:
            if from_node in id_to_crop_object_mapping[to_node].inlinks:
                logging.info('Adding edge that is alredy in the graph: {} --> {}'
                             ' -- doing nothing'.format(from_node, to_node))
                return
            else:
                raise NotationGraphError('Found {0} in outlinks of {1}, but not {1} in inlinks of {0}!'
                                         ''.format(to_node, from_node))
        elif from_node in id_to_crop_object_mapping[to_node].inlinks:
            raise NotationGraphError('Found {0} in inlinks of {1}, but not {1} in outlinks of {0}!'
                                     ''.format(from_node, to_node))

        id_to_crop_object_mapping[from_node].outlinks.append(to_node)
        id_to_crop_object_mapping[to_node].inlinks.append(from_node)

    def model_output_to_midi(self, output_repr):
        return midi_matrix_to_midi(output_repr)


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--model', default="base_convnet",
                        help='The name of the model that you wish to use.')
    parser.add_argument('-p', '--params', default="models/default_model.tsd",
                        help='The exported model from the training.')
    parser.add_argument('-c', '--config', default="exp_configs/muscima_bboxes.yaml",
                        help='The config file that controls how inputs to the network will be extracted from MuNGOs.')
    parser.add_argument('-i', '--input_image', required=True,
                        help='A single-system input image for which MIDI should'
                             ' be output. This is the simplest input mode.'
                             ' If a directory is given, it will run over all'
                             ' images in that directory, expecting --input_mung'
                             ' to also be a directory with correspondingly named'
                             ' MuNG files (like for training).')
    parser.add_argument('-g', '--input_mung', required=True,
                        help='A MuNG XML file. The edges inlinks/outlinks in'
                             ' the file are ignored. If this is a'
                             ' directory, it will run over all MuNGs in that'
                             ' directory, expecting --input_image to also'
                             ' be a directory with correspondingly named'
                             ' image files (like for training).')
    parser.add_argument('-o', '--output_mung_directory', required=True,
                        help='The directory that will contain the MuNGs.')
    parser.add_argument('--batch_size', type=int, action='store', default=10,
                        help='The runtime iterator batch size.')
    parser.add_argument('--mock', action='store_true',
                        help='If set, will not load a real model and just run'
                             ' a mock prediction using MockNetwork.predict()')
    parser.add_argument('--play', action='store_true',
                        help='If set, will run MIDI inference over the output'
                             ' MuNG and play the result.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')

    return parser


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info('Starting main...')
    start_time = time.time()

    logging.info('Loading config: {}'.format(args.config))
    config = load_config(args.config)

    logging.info('Loading model: {}'.format(args.model))
    mung_linker_network = select_model(args.model, args.batch_size)
    runtime_batch_iterator = mung_linker_network.runtime_batch_iterator()

    if args.mock:
        logging.info('Using mock network, so no parameters will be loaded.')
        model = MockNetwork(args.batch_size)
    else:
        logging.info('Loading model checkpoint from state dict: {0}'.format(args.params))
        checkpoint = torch.load(args.params)
        mung_linker_network.load_state_dict(checkpoint['model_state_dict'])
        model = PyTorchNetwork(net=mung_linker_network)

    runner = MunglinkerRunner(model=model,
                              config=config,
                              runtime_batch_iterator=runtime_batch_iterator)

    image_files = []
    input_mung_files = []

    if os.path.isfile(args.input_image):
        logging.info('Loading image: {}'.format(args.input_image))
        image_files.append(args.input_image)
    elif os.path.isdir(args.input_image):
        image_files.extend(glob(args.input_image + "/*.png"))

    if os.path.isfile(args.input_mung):
        logging.info('Loading MuNG: {}'.format(args.input_mung))
        input_mung_files.append(args.input_mung)
    elif os.path.isdir(args.input_mung):
        input_mung_files.extend(glob(args.input_mung + "/*.xml"))

    output_mung_files = [os.path.join(args.output_mung_directory, os.path.basename(f)) for f in input_mung_files]
    os.makedirs(args.output_mung_directory, exist_ok=True)

    if len(image_files) != len(input_mung_files):
        raise Exception("Length of images and MuNGs is not the same")

    for i, (image_file, input_mung_file, output_mung_file) in enumerate(
            zip(image_files, input_mung_files, output_mung_files)):
        input_mungos = parse_cropobject_list(input_mung_file)
        input_mung = NotationGraph(input_mungos)

        print('Running Munglinker: {} / {}'.format(i, len(image_files)))
        output_mung = runner.run(image_file, input_mung)
        with open(output_mung_file, 'w') as file:
            file.write(export_cropobject_list(output_mung.cropobjects))

        precision, recall, f1_score, true_positives, false_positives, false_negatives = \
            evaluate_result(input_mung_file, output_mung_file)

        if args.play:
            mf = build_midi(cropobjects=output_mung.cropobjects)
            with open("output.midi", 'wb') as stream_out:
                 mf.writeFile(stream_out)

    end_time = time.time()
    logging.info('run.py done in {0:.3f} s'.format(end_time - start_time))
