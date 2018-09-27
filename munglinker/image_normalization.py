"""This module implements image normalization: all the processes
used to ensure that the model is getting the inputs which it is
expecting."""
from __future__ import print_function, unicode_literals
import logging
import time

import numpy

from skimage.filters import gaussian, threshold_otsu

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class ImageNormalizer:
    """Wrapper class that ensures image normalization."""
    def __init__(self, do_auto_invert, do_stretch_intensity, do_binarize):
        self.auto_invert = do_auto_invert
        self.stretch_intensity = do_stretch_intensity
        self.binarize = do_binarize

    def process(self, image):
        if self.auto_invert:
            image = auto_invert(image, smoothing_sigma=2.0)
        if self.stretch_intensity:
            image = stretch_intensity(image, smoothing_sigma=2.0)
        if self.binarize:
            image = binarize_nnz(image)

        return image


def auto_invert(image, smoothing_sigma=2.0):
    """Attempts to make sure that foreground is white and background
    is black.

    Assumes that the background has many more pixels than the foreground.

    * Finds maximum and minimum intensities (after some smoothing, to
      minimize speckles/salt and pepper noise)
    * Finds median intensity
    * If the median is closer to maximum, then assumes the image
      has a light background and should be inverted.
    * If the median is closer to minimum, then assumes the image
      has a dark background and should NOT be inverted.
    """
    output = image * 1
    blurred = gaussian(output, sigma=smoothing_sigma)

    i_max = blurred.max()
    i_min = blurred.min()
    i_med = numpy.median(blurred)

    if (i_max - i_med) < (i_med - i_min):
        output = numpy.invert(output)

    return output


def stretch_intensity(image, smoothing_sigma=2.0):
    """Stretches the intensity range of the image to (0, 255)."""
    output = image * 1
    blurred = (gaussian(output, sigma=smoothing_sigma) * 255).astype('uint8')

    i_max = blurred.max()
    i_min = blurred.min()

    logging.info('Stretching image intensity: min={0}, max={1}'
                 ''.format(i_min, i_max))

    output[output > i_max] = i_max
    output[output < i_min] = i_min

    output -= i_min
    output = (output * (255 / i_max)).astype('uint8')

    return output


def binarize_nnz(image, foreground_intensity=255):
    # Is image already binary?
    values = image.ravel()[numpy.flatnonzero(image)]
    try:
        nnz_threshold = threshold_otsu(values)
    except ValueError:
        # It's already binary.
        return image
    output = image * 1
    output[output < nnz_threshold] = 0
    output[output >= nnz_threshold] = foreground_intensity
    return output


def rebinarize_nnz(image):
    threshold = threshold_otsu(image)
    b_image = image * 1
    b_image[b_image < threshold] = 0

    rb_image = binarize_nnz(b_image)
    return rb_image