import argparse
import os
import shutil
from distutils.dir_util import copy_tree

from omrdatasettools.Downloader import Downloader, OmrDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--mung_root', action='store', default="../data/mungs",
                        help='The root directory that will contain the MuNG XMLs.')
    parser.add_argument('-i', '--image_root', action='store', default="../data/images",
                        help='The root directory that will contain the images of'
                             ' scores that are represented by the MuNGs. The'
                             ' image names must correspond to the MuNG file'
                             ' names, up to the file type suffix.')
    args = parser.parse_args()
    mung_root_directory = args.mung_root
    image_root_directory = args.image_root

    os.makedirs(mung_root_directory, exist_ok=True)
    os.makedirs(image_root_directory, exist_ok=True)

    temporary_directory = "temp"
    dataset_downloader = Downloader()
    dataset_downloader.download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_V2, temporary_directory)

    copy_tree(os.path.join(temporary_directory, "v2.0", "data", "annotations"), mung_root_directory)
    copy_tree(os.path.join(temporary_directory, "v2.0", "data", "images"), image_root_directory)

    shutil.rmtree(temporary_directory)
