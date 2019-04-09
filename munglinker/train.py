import argparse
import logging
import time

import torch
from torch.nn import BCELoss
import numpy as np

from munglinker.data_pool import load_munglinker_data
from munglinker.losses import FocalLoss
from munglinker.model import PyTorchNetwork
from munglinker.training_strategies import PyTorchTrainingStrategy
from munglinker.utils import build_experiment_name, select_model
from torchsummary import summary


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--model', default="base_convnet",
                        help='The name of the model that you wish to use. '
                             'Must be one of ["base_convnet", "base_convnet_double_filters"].')
    parser.add_argument('--continue_training', action='store_true',
                        help='If set, checks whether a model under the name set'
                             ' in -e already exists. If it does, initialize training'
                             ' using its parameters.')
    parser.add_argument('-r', '--mung_root', action='store', default="data/mungs",
                        help='The root directory that contains the MuNG XMLs.')
    parser.add_argument('-i', '--image_root', action='store', default="data/images",
                        help='The root directory that contains the images of'
                             ' scores that are represented by the MuNGs. The'
                             ' image names must correspond to the MuNG file'
                             ' names, up to the file type suffix.')
    parser.add_argument('-s', '--split_file', action='store', default="splits/mob_split.yaml",
                        help='The split file that specifies which MUSCIMA++ items'
                             ' are training, validation, and test data. See the'
                             ' splits/ subdirectory for examples.')

    parser.add_argument('-c', '--config_file', action='store', default="exp_configs/muscima_bboxes.yaml",
                        help='The config file that specifies things like'
                             ' preprocessing. See the exp_configs/ subdirectory'
                             ' for examples.')

    parser.add_argument('-e', '--export', action='store', default="models/default_model.tsd",
                        help='Export the model params into this file.')

    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='Minibatch size for training.')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of steps without improvement in validation'
                             ' loss after which the learning rate is attenuated.')

    parser.add_argument('--n_epochs_per_checkpoint', type=int, default=1,
                        help='Make a checkpoint of the model every N epochs.'
                             ' The checkpoint goes under the same name as -e.')
    parser.add_argument('--initial_learning_rate', type=float, default=0.001,
                        help='Sets the initial learning rate for the optimizer')

    parser.add_argument('-a', '--augmentation', action='store_true',
                        help='If set, will train with data augmentation:'
                             ' scaling magnitude 0.4, rotation 0.2,'
                             ' vertical dilation 4, horizontal dilation 1')

    parser.add_argument('--exp_tag', action='store',
                        help='Give the experiment some additional name.')

    parser.add_argument('--tensorboard_log_dir', default="logging", help='Tensoroboard logs directory.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    _start_time = time.time()

    mung_linker_network = select_model(args.model, args.batch_size)

    print(mung_linker_network)
    summary(mung_linker_network, (3, 256, 512), device="cpu")

    loss_function = BCELoss()

    exp_name = build_experiment_name(args)

    checkpoint_export_file = args.export + '.ckpt'

    strategy = PyTorchTrainingStrategy(name=exp_name,
                                       loss_function=loss_function,
                                       n_epochs_per_checkpoint=args.n_epochs_per_checkpoint,
                                       best_model_by_fscore=False,
                                       max_epochs=args.n_epochs,
                                       batch_size=args.batch_size,
                                       checkpoint_export_file=checkpoint_export_file,
                                       best_params_file=args.export,
                                       improvement_patience=args.patience,
                                       initial_learning_rate=args.initial_learning_rate)

    print(strategy.summary())

    model = PyTorchNetwork(mung_linker_network, strategy, args.tensorboard_log_dir)
    initial_epoch = 1
    previously_best_validation_loss = np.inf

    if args.continue_training:
        try:
            checkpoint = torch.load(checkpoint_export_file)
            model.net.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initial_epoch = checkpoint['epoch'] + 1
            previously_best_validation_loss = checkpoint['best_validation_loss']
            print('Loaded model from checkpoint: {0}. Resuming at epoch {1}'.format(checkpoint_export_file,
                                                                                    initial_epoch))
        except OSError as e:
            print('Error during loading of previously saved checkpoint {0}: {1}'.format(checkpoint_export_file, e))

    data = load_munglinker_data(
        mung_root=args.mung_root,
        images_root=args.image_root,
        split_file=args.split_file,
        config_file=args.config_file,
        load_training_data=True,
        load_validation_data=True,
        load_test_data=False
    )
    print('Loaded pools; training data has {} entities'.format(len(data['train'].train_entities)))

    train_batch_iter = mung_linker_network.train_batch_iterator()
    valid_batch_iter = mung_linker_network.valid_batch_iterator()
    test_batch_iter = mung_linker_network.test_batch_iterator()
    batch_iters = {'train': train_batch_iter, 'valid': valid_batch_iter, 'test': test_batch_iter}

    print('Data initialized.')

    model.fit(data=data,
              batch_iters=batch_iters,
              dump_file=None,
              log_file=None,
              initial_epoch=initial_epoch,
              previously_best_validation_loss=previously_best_validation_loss)

    print('Saving model to: {0}'.format(args.export))
    torch.save(mung_linker_network.state_dict(), args.export)

    _end_time = time.time()
    print('train.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
