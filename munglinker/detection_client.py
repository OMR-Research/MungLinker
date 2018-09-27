#!/usr/bin/env python
"""This is a script that..."""
from __future__ import print_function, unicode_literals
import argparse
import copy
import logging
import os
import socket
import time
import uuid

import shutil

import itertools

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

import numpy
import pickle
import scipy.misc

from muscima.io import parse_cropobject_list, export_cropobject_list

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


CLASSES=["staff_line",
         "notehead-full",
         "ledger_line",
         "sharp",
         "flat",
         "natural",
         "stem",
         "beam",
         "duration-dot",
         "g-clef",
         "f-clef",
         "c-clef",
         "quarter_rest",
         "half_rest",
         "whole_rest",
         "8th_rest",
         "8th_flag",
         "measure_separator"]


# Copied over from muscima.cropobject, because fixing it there did not seem to work
def merge_cropobject_lists(*cropobject_lists):
    """Combines the CropObject lists from different documents
    into one list, so that inlink/outlink references still work.
    This is useful only if you want to merge two documents
    into one (e.g., if your annotators worked on different "layers"
    of data, and you want to merge these annotations).

    This just means shifting the ``objid`` (and thus inlinks
    and outlinks). It is assumed the lists pertain to the same
    image. Uses deepcopy to avoid exposing the original lists
    to modification through the merged list.

    .. warning::

        If you are ever exporting the merged list, make sure to
        set the ``uid`` for the outputs correctly, if you want
        to create a new document.

    .. warning::

        Currently cannot handle precedence edges.

    """
    max_objids = [max([c.objid for c in c_list]) for c_list in cropobject_lists]
    min_objids = [min([c.objid for c in c_list]) for c_list in cropobject_lists]
    shift_by = [0] + [sum(max_objids[:i]) - min_objids[i] + 1 for i in range(1, len(max_objids))]

    print('max_objids: {}'.format(max_objids))
    print('min_objids: {}'.format(min_objids))
    print('shift_by: {}'.format(shift_by))

    new_lists = []
    for clist, s in zip(cropobject_lists, shift_by):
        new_list = []
        for c in clist:
            new_c = copy.deepcopy(c)
            # UID handling
            collection, doc, _ = new_c.parse_uid()
            new_uid = new_c.build_uid(collection, doc, c.objid + s)
            new_objid = c.objid + s
            new_c.objid = new_objid
            new_c.set_uid(new_uid)

            # Graph handling
            new_c.inlinks = [i + s for i in c.inlinks]
            new_c.outlinks = [o + s for o in c.outlinks]

            # Should also handle precedence...?

            new_list.append(new_c)
        new_lists.append(new_list)

    output = list(itertools.chain(*new_lists))

    return output


##############################################################################


class ObjectDetectionOMRAppClient(object):
    """Handles the client-side networking for object
    detection. Very lightweight -- only builds the socket,
    sends the request, receives the response and writes it
    to the file specified by ObjectDetectionHandler.

    Can be therefore used outside MUSCIMarker.

    Not a Kivy widget."""
    def __init__(self, host, port, tmp_dir):
        self.host = host
        self.port = port

        self.BUFFER_SIZE = 256
        self.tmp_dir = tmp_dir

    def _format_request(self, request):
        f_request = dict()
        for k in request:
            if k == 'image???':
                f_request[k] = self._format_request_image(request[k])
            else:
                f_request[k] = request[k]
        return f_request

    def _format_request_image(self, image):
        return numpy.ndarray.dumps(image)

    def call(self, request):
        logging.info('ObjectDetectionOMRAppClient.run(): starting')
        _start_time = time.clock()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = '127.0.0.1'  # socket.gethostname()

        logging.info('ObjectDetectionOMRAppClient: connecting to host {0}, port {1}'
                     ''.format(host, self.port))
        s.connect((host, self.port))
        logging.info('ObjectDetectionOMRAppClient.run(): connected to'
                     ' host {0}, port {1}'.format(host, self.port))

        # Format request and dump to request file
        f_request = self._format_request(request)
        _rstring = str(uuid.uuid4())
        temp_basename = 'fcnomr-detection_client.request.' + _rstring + '.pkl'
        request_fname = os.path.join(self.tmp_dir, temp_basename)
        with open(request_fname, 'w') as fh:
            pickle.dump(f_request, fh, protocol=0)

        with open(request_fname, 'rb') as fh:
            data = fh.read(self.BUFFER_SIZE)
            _n_data_packets_sent = 0
            while data:
                if _n_data_packets_sent % 5000 == 0:
                    logging.info('ObjectDetectionOMRAppClient.run(): sending data,'
                                 'iteration {0}'.format(_n_data_packets_sent))
                s.send(data)
                data = fh.read(self.BUFFER_SIZE)
                _n_data_packets_sent += 1

        # s.send(b"Hello server!")
        logging.info('Shutting down socket for writing...')
        s.shutdown(socket.SHUT_WR)

        logging.info('Finished sending, waiting to receive.')
        # Server does its thing now. We wait at s.recv()

        response_basename = 'MUSCIMarker.omrapp-response.' + _rstring + '.xml'
        response_fname = os.path.join(self.tmp_dir, response_basename)

        _n_data_packets_received = 0
        with open(response_fname, 'wb') as f:
            _n_data_packets_received = 0
            logging.info('file opened: {0}'.format(response_fname))
            while True:
                if _n_data_packets_received % 1000 == 0:
                    logging.info('ObjectDetectionOMRAppClient.run(): receiving data,'
                                 'iteration {0}'.format(_n_data_packets_received))
                data = s.recv(self.BUFFER_SIZE)

                if not data:
                    break
                f.write(data)

                _n_data_packets_received += 1
        logging.info('Received data: {} packets'.format(_n_data_packets_received))
        f.close()

        try:
            cropobjects = parse_cropobject_list(response_fname)
            # print(export_cropobject_list(cropobjects))
            # Verify that result is valid (re-request on failure?)
            if os.path.isfile(response_fname):
                os.unlink(response_fname)

        except:
            logging.warning('ObjectDetectionHandler: Could not parse'
                            ' response file {0}'.format(response_fname))
            cropobjects = []
        finally:
            # Cleanup files.
            logging.info('Cleaning up temp files.')
            if os.path.isfile(request_fname):
                os.unlink(request_fname)

        logging.info('Successfully got the file')

        s.close()
        logging.info('connection closed')

        del s

        _end_time = time.clock()
        logging.info('fcnomr.detection_client.ObjectDetectionOMRAppClient.run():'
                     ' done in {0:.3f} s'.format(_end_time - _start_time))

        return self.postprocess_cropobjects(cropobjects)


    def postprocess_cropobjects(self, cropobjects):
        """Handler-specific CropObject postprocessing. Can be configurable
        through MUSCIMarker settings."""
        filtered_cropobjects = filter_small_cropobjects(cropobjects)
        filtered_cropobjects = filter_thin_cropobjects(filtered_cropobjects)
        return filtered_cropobjects



def filter_small_cropobjects(cropobjects, threshold=20):
    return [c for c in cropobjects if c.mask.sum() > threshold]


def filter_thin_cropobjects(cropobjects, threshold=2):
    return [c for c in cropobjects if min(c.width, c.height) > threshold]



##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--img', action='store', required=True,
                        help='Load the image from this file.')
    parser.add_argument('-s', '--stafflines', action='store',
                        help='Read stafflines MuNG from this file.')
    parser.add_argument('-o', '--output_mung', action='store', default=None,
                        help='Write the resulting MuNG to this'
                             ' file.')

    parser.add_argument('-p', '--port', type=int, action='store', default=33554,
                        help='Port to which to connect on localhost. (Relies'
                             ' on SSH tunnels to communicate with a detection'
                             ' server not running locally.)')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    logging.info('Initializing detection client.')
    tmp_dir = 'fcnomr-detection_client_{}.tmp'.format(uuid.uuid4())
    os.mkdir(tmp_dir)
    client = ObjectDetectionOMRAppClient(host=None,
                                         port=args.port,
                                         tmp_dir=tmp_dir)

    logging.info('Loading image.')
    image = scipy.misc.imread(args.img, mode='L')

    logging.info('Loading staffline annotations: {}'.format(args.stafflines))
    stafflines = []
    if args.stafflines:
        stafflines = parse_cropobject_list(args.stafflines)

    # Run detection
    logging.info('Building detection request')
    request = {'image': image,
               'clsname': CLASSES,
               }
    logging.info('Calling detection client.')
    try:
        detected_cropobjects = client.call(request)
    except socket.error:
        raise

    logging.info('Merging stafflines and detected cropobjects.')
    output_cropobjects = merge_cropobject_lists(detected_cropobjects, stafflines)

    docname = os.path.splitext(os.path.basename(args.output_mung))[0]
    logging.info('Exporting cropobject list with docname {} to {}'.format(docname, args.output_mung))
    xml = export_cropobject_list(output_cropobjects,
                                 docname=docname,
                                 dataset_name='FNOMR_results')
    with open(args.output_mung, 'wb') as out_stream:
        out_stream.write(xml)
        out_stream.write('\n')

    logging.info('Cleaning up temp files')
    shutil.rmtree(tmp_dir)
    if os.path.isdir(tmp_dir):
        os.rmdir(tmp_dir)

    _end_time = time.clock()
    logging.info('detection_client.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
