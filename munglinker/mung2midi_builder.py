#!/usr/bin/env python
"""This is a script that builds the MIDI file out of a notation
graph, given that the graph is sufficient for building the MIDI."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import os
import time
import traceback
from typing import List

from mung import constants
from mung.graph import NotationGraph
from mung.io import read_nodes_from_file

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################
# MIDI export
from mung.node import Node
from mung2midi.inference import PitchInferenceEngine, OnsetsInferenceEngine, MIDIBuilder


def build_midi(nodes: List[Node], selected_nodes=None,
               retain_pitches=True,
               retain_durations=True,
               retain_onsets=True,
               tempo=180):
    """Attempts to export a MIDI file from the current graph. Assumes that
    all the staff objects and their relations have been correctly established,
    and that the correct precedence graph is available.

    :param retain_pitches: If set, will record the pitch information
        in pitched objects.

    :param retain_durations: If set, will record the duration information
        in objects to which it applies.

    :returns: A single-track ``midiutil.MidiFile.MIDIFile`` object. It can be
        written to a stream using its ``mf.writeFile()`` method."""
    _cdict = {c.id: c for c in nodes}

    pitch_inference_engine = PitchInferenceEngine()
    time_inference_engine = OnsetsInferenceEngine(nodes=nodes)

    try:
        logging.info('Running pitch inference.')
        pitches, pitch_names = pitch_inference_engine.infer_pitches(nodes,
                                                                    with_names=True)
    except Exception as e:
        logging.warning('Model: Pitch inference failed!')
        logging.exception(traceback.format_exc(e))
        raise

    if retain_pitches:
        for objid in pitches:
            c = _cdict[objid]
            pitch_step, pitch_octave = pitch_names[objid]
            c.data['midi_pitch_code'] = pitches[objid]
            c.data['normalized_pitch_step'] = pitch_step
            c.data['pitch_octave'] = pitch_octave

    try:
        logging.info('Running durations inference.')
        durations = time_inference_engine.durations(nodes)
    except Exception as e:
        logging.warning('Model: Duration inference failed!')
        logging.exception(traceback.format_exc(e))
        raise

    if retain_durations:
        for objid in durations:
            c = _cdict[objid]
            c.data['duration_beats'] = durations[objid]

    try:
        logging.info('Running onsets inference.')
        onsets = time_inference_engine.onsets(nodes)
    except Exception as e:
        logging.warning('Model: Onset inference failed!')
        logging.exception(traceback.format_exc(e))
        raise

    if retain_onsets:
        for objid in onsets:
            c = _cdict[objid]
            c.data['onset_beats'] = onsets[objid]

    # Process ties
    durations, onsets = time_inference_engine.process_ties(nodes,
                                                           durations, onsets)

    # Prepare selection subset
    if selected_nodes is None:
        selected_nodes = _cdict.values()
    selection_objids = [c.objid for c in selected_nodes]

    # Build the MIDI data
    midi_builder = MIDIBuilder()
    mf = midi_builder.build_midi(
        pitches=pitches, durations=durations, onsets=onsets,
        selection=selection_objids, tempo=tempo)

    return mf


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input_mung', action='store', required=True,
                        help='Load the MuNG from this file.')
    parser.add_argument('-o', '--output_midi', action='store', default=None,
                        help='Write the resulting MIDI to this'
                             ' file.')

    parser.add_argument('--output_dir', action='store',
                        help='Write the resulting MIDI files into this'
                             ' directory. (Should exist.) If this option'
                             ' is used, the output name will be derived'
                             ' from the --input_mung basename.')
    parser.add_argument('--per_staff', action='store_true',
                        help='If set, will export a separate MIDI file'
                             ' for each staff. The filename includes the'
                             ' string "staff" and the corresponding objid.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.time()

    nodes = read_nodes_from_file(args.input_mung)
    graph = NotationGraph(nodes)

    ######################################################################
    # export MIDI for each staff separately
    if args.per_staff:
        if not args.output_dir:
            raise ValueError('When exporting per-staff MIDI, the output'
                             ' directory must be specified.')

        basename = os.path.splitext(os.path.basename(args.input_mung))[0]
        # ...store all the output files in the same args.output_dir,
        #    so that it is easier to just load them all in midi-evaluation.py
        # output_path = os.path.join(output_path, basename)
        # if not os.path.isdir(output_path):
        #     os.mkdir(output_path)

        # basename is used both in the output dir, for grouping the per-staff
        # MIDI, and in the names of the MIDI files themselves.
        _output_base = os.path.join(args.output_dir, basename)

        # Collect all staffs.
        # They are sorted top-down, so that during retrieval, we can easily
        # check for a hit.
        staffs = sorted([c for c in nodes if c.class_name == 'staff'],
                        key=lambda x: x.top)

        # For each staff, collect its noteheads
        noteheads_per_staff = {
            s.id: graph.parents(s, class_filter=constants.InferenceEngineConstants.NOTEHEAD_CLASS_NAMES)
            for s in staffs
        }

        for staff_idx, s in enumerate(staffs):
            noteheads = noteheads_per_staff[s.id]
            if len(noteheads) == 0:
                continue
            logging.info('Processing staff: {0}, noteheads: {1}'
                         ''.format(s.id, len(noteheads)))

            mf = build_midi(nodes, selected_nodes=noteheads)

            output_path = _output_base + '.staff-{0}'.format(staff_idx) + '.mid'
            with open(output_path, 'wb') as stream_out:
                mf.writeFile(stream_out)

    ###################################################################
    # straightforward single-file processing
    else:
        output_path = args.output_midi
        if args.output_dir:
            basename = os.path.splitext(os.path.basename(args.input_mung))[0]
            output_path = os.path.join(args.output_dir, basename + '.mid')

        mf = build_midi(nodes=nodes)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as stream_out:
            mf.writeFile(stream_out)

    _end_time = time.time()
    logging.info('mung2midi_builder.py done in {0:.3f} s'.format(_end_time - _start_time))


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
