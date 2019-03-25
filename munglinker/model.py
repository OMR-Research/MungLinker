"""This file defines the pytorch fit() wrapper."""
from __future__ import print_function, unicode_literals, division

import collections
import logging
import os
import pprint
import sys
import time

import numpy
import numpy.random
import torch
from torch.autograd import Variable
from tqdm import tqdm

from munglinker.batch_iterators import threaded_generator_from_iterator
from munglinker.evaluation import eval_clf_by_class_pair, print_class_pair_results
from munglinker.utils import ColoredCommandLine, targets2classes

torch.set_default_tensor_type('torch.FloatTensor')

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class PyTorchNetwork(object):
    """The PyTorchNetwork class wraps a PyTorch ``nn.Module`` instance
    with a sklearn-like interface (fit, predict, predict_proba, etc.)
    It should be universal: independent of what net it is wrapping, and
    what task this net is used for; therefore, you have to make sure that
    you are feeding the correct data to the correct model, and that the
    training strategy is correct.

    The biggest workhorse is the ``fit`` function.

    Adapted from lasagne_wrapper.network.Network by Matthias Dorfer
    (matthias.dorfer@jku.at).
    """

    def __init__(self, net):
        self.net = net
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.net.cuda()

        # Logging to tesnorboard through here
        self.tensorboard = None

    def predict(self,
                data_pool,
                runtime_batch_iterator):
        """Runs the model prediction. Expects a data pool and a runtime
        batch iterator.

        :param data_pool: The runtime data pool. It still produces targets
            (these will be 0 in most cases...), but they are ignored. The
            correct settings for a runtime pool are to (1) set negative sample
            max to -1, so that all candidates within the threshold distance
            get classified, (2) use a grammar to restrict candidate pairs,
            (3) do *NOT* resample after epoch end.

        :param runtime_batch_iterator: Batch iterator from a model's runtime.

        :returns: A list of (mungo_pairs, predictions). After applying this
            method, you will have to take care of actually adding the predicted
            edges into the graphs -- split the MuNG pairs by documents, etc.
        """
        # Ensure correct data pool behavior.
        data_pool.resample_train_entities = False

        # Initialize data feeding from iterator
        iterator = runtime_batch_iterator(data_pool)
        generator = threaded_generator_from_iterator(iterator)

        n_batches = len(data_pool) // runtime_batch_iterator.batch_size
        print('n. of runtime entities: {}; batches: {}'
              ''.format(len(data_pool), n_batches))

        # Aggregate results (two-way)
        all_mungo_pairs = []
        all_np_predictions = numpy.zeros((0, 2))

        # Run generator
        for batch_idx, _data_point in enumerate(generator):
            mungo_pairs, np_inputs = _data_point  # next(generator)

            mungo_pairs = list(mungo_pairs)
            all_mungo_pairs.extend(mungo_pairs)

            inputs = self.__np2torch(np_inputs)
            predictions = self.net(inputs)
            np_predictions = self.__torch2np(predictions)
            all_np_predictions = numpy.concatenate((all_np_predictions, np_predictions))

        all_np_predicted_classes = targets2classes(all_np_predictions)
        logging.info('Prediction: {} out of {} positive'
                     ''.format(all_np_predicted_classes.sum(), all_np_predicted_classes.size))
        return all_mungo_pairs, all_np_predicted_classes

    def fit(self,
            data,
            batch_iters,
            training_strategy,
            dump_file=None,
            log_file=None,
            tensorboard_log_path=None):
        """Trains the model.

        :param data: A dict that has a ``'train'`` and a ``'valid'`` key,
            and its values are objects that implement the ``DataPool``
            interface for ``__getitem__``.

        :param batch_iters: A dict that has a ``train`` and ``valid`` key,
            and its values are objects

        :param training_strategy: A ``NamedTuple``-like class that aggregates
            parameters of the training process: optimizer type, initial LR and
            decay, loss function, etc.

        :param dump_file: The trained net's state dict will be dumped
            into this file. The dump happens every ecpoch in which the model
            improves, so that the best trained model is always saved.

        :param log_file: Training progress will be logged to this file.

        :param tensorboard_log_path: TensorBoard writer will write to this
            directory.
        """
        print("Training neural network...")
        colored_command_line = ColoredCommandLine()

        if dump_file is not None:
            out_path = os.path.dirname(dump_file)
            if out_path and not os.path.isdir(out_path):
                os.mkdir(out_path)

        if log_file is not None:
            out_path = os.path.dirname(log_file)
            if out_path and not os.path.isdir(out_path):
                os.mkdir(out_path)

        if tensorboard_log_path is not None:
            out_path = os.path.dirname(tensorboard_log_path)
            if out_path and not os.path.isdir(out_path):
                os.mkdir(out_path)
            from tensorboardX import SummaryWriter
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_path,
                                                          training_strategy.name),
                                             comment=training_strategy.name)

        # Extra variable for learning rate, since it attenuates during training
        learning_rate = training_strategy.ini_learning_rate

        # We don't need to create iter functions.

        # Initialize evaluation outputs
        training_results, validation_results = [], []

        # Train and validation loss, for early stopping
        training_losses, validation_losses = [numpy.inf], [numpy.inf]

        # Early stopping
        epochs_since_last_improvement = 0
        best_model = self.net.state_dict()

        # Refinement
        refinement_stage = False
        refinement_steps = 0

        # Tracking
        best_loss, best_training_loss, best_validation_loss = 1e7, 1e7, 1e7
        previous_fscore_training, previous_fscore_validation = 0.0, 0.0

        print("Starting training...")
        try:

            ##################################################################
            # Preparation
            if training_strategy.best_model_by_fscore:
                best_loss = previous_fscore_training

            # Set batch sizes.
            data['train'].batch_size = training_strategy.batch_size

            # Initialize loss and optimizer
            loss_fn = training_strategy.init_loss_fn()
            optimizer = training_strategy.optimizer_class(self.net.parameters(),
                                                          lr=learning_rate)

            ##################################################################
            # Iteration
            for current_epoch_index in range(training_strategy.max_epochs):

                ##################################################################
                # Train the epoch
                training_epoch_output = self.__train_epoch(data['train'],
                                                           batch_iters['train'],
                                                           loss_fn,
                                                           optimizer)
                training_results.append(training_epoch_output)
                training_loss = training_epoch_output['train_loss']
                training_losses.append(training_loss)

                # Checkpointing
                if (current_epoch_index + 1) % training_strategy.n_epochs_per_checkpoint == 0:
                    if training_strategy.checkpoint_export_file:
                        if training_strategy.checkpoint_export_file is None:
                            os.makedirs("models", exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(training_strategy.checkpoint_export_file),
                                        exist_ok=True)
                        torch.save(self.net.state_dict(),
                                   training_strategy.checkpoint_export_file)
                    else:
                        logging.warning('Cannot checkpoint: no checkpoint file'
                                        ' specified in training strategy!')

                ##################################################################
                # Validation: run with detector on checkpoints, otherwise
                # only collect loss.
                print('\nValidating epoch {}'.format(current_epoch_index))

                #########################################################
                # This only happens once per n_epochs_per_checkpoint
                if (current_epoch_index + 1) % training_strategy.n_epochs_per_checkpoint == 0:
                    validation_epoch_output = self.__validate_epoch(data['valid'],
                                                                    batch_iters['valid'],
                                                                    loss_fn, training_strategy)
                    validation_results.append(validation_epoch_output)

                    self.__log_epoch_to_tensorboard(current_epoch_index,
                                                    training_epoch_output,
                                                    validation_epoch_output)

                    if 'fsc' in validation_epoch_output:
                        validation_loss = -1 * validation_epoch_output['all']['fsc'][-1]
                    else:
                        validation_loss = validation_epoch_output['all']['loss']
                    validation_losses.append(validation_loss)

                    print('Validation results: {}'.format(pprint.pformat(validation_epoch_output['all'])))

                    print_class_pair_results(validation_epoch_output)

                ##################################
                # Log validation loss

                epoch_loss = validation_losses[-1]
                validation_loss_string = '\tValidation loss: {0:.6f}\t(Patience: {1})' \
                                         ''.format(epoch_loss,
                                                   training_strategy.improvement_patience - epochs_since_last_improvement)
                if epoch_loss < best_loss:
                    colored_command_line.print(validation_loss_string, ColoredCommandLine.OKGREEN)
                else:
                    print(validation_loss_string)

                ##############################################################
                # Early-stopping: Check for improvement
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = self.net.state_dict()
                    # Save the best model!
                    if training_strategy.best_params_file:
                        print('Saving best model: {}'
                              ''.format(training_strategy.best_params_file))
                        torch.save(self.net.state_dict(),
                                   training_strategy.best_params_file)
                    else:
                        logging.warning('No file to save the best model is specified'
                                        ' in training strategy!!!')

                    epochs_since_last_improvement = 0
                else:
                    epochs_since_last_improvement += 1

                # Early-stopping: continue, refine, or end training
                if (refinement_stage and (epochs_since_last_improvement > training_strategy.refinement_patience)) \
                        or (epochs_since_last_improvement > training_strategy.improvement_patience):
                    print('Early-stopping: exceeded patience in epoch {0}'
                          ''.format(current_epoch_index))

                    epochs_since_last_improvement = 0
                    refinement_stage = True
                    if refinement_steps < training_strategy.n_refinement_steps:
                        self.__update_learning_rate(optimizer,
                                                    training_strategy.lr_refinement_multiplier)
                        refinement_steps += 1

                    else:
                        print('Early-stopping: exceeded refinement budget,'
                              ' training ends.')

                        print('---------------------------------')
                        print('Final validation:\n')

                        validation_epoch_output = self.__validate_epoch(data['valid'], batch_iters['valid'], loss_fn,
                                                                        training_strategy)
                        validation_results.append(validation_epoch_output)

                        self.__log_epoch_to_tensorboard(current_epoch_index,
                                                        training_epoch_output,
                                                        validation_epoch_output)

                        break

        except KeyboardInterrupt:
            pass

        # Set net to best weights
        self.net.load_state_dict(best_model)

        # Export the best weights
        if training_strategy.best_params_file:
            torch.save(self.net.state_dict(),
                       training_strategy.best_params_file)
        else:
            logging.warning('No file to save the best model is specified'
                            ' in training strategy!!!')

        # Return best validation loss
        if training_strategy.best_model_by_fscore:
            return best_loss * -1
        else:
            return best_loss

    def __log_epoch_to_tensorboard(self, epoch_index,
                                   training_epoch_outputs,
                                   validation_epoch_outputs):

        if self.tensorboard is None:
            return

        self.tensorboard.add_scalar('train/loss',
                                    training_epoch_outputs['train_loss'], epoch_index)

        for label in validation_epoch_outputs:
            if label == 'all':
                continue
            if len(label) == 2:
                label_name = '{}__{}'.format(label[0], label[1])
            else:
                label_name = str(label)
            for k, v in validation_epoch_outputs[label].items():
                self.tensorboard.add_scalar('{0}/{1}'.format(label_name, k),
                                            v, epoch_index)

        print(validation_epoch_outputs['all'])
        for k, v in validation_epoch_outputs['all'].items():
            try:
                self.tensorboard.add_scalar('{0}'.format(k, v, epoch_index),
                                            v, epoch_index)
            except AssertionError:
                self.tensorboard.add_scalar('{0}'.format(k, v, epoch_index),
                                            v[-1], epoch_index)

    def __validate_epoch(self, data_pool, validation_batch_iterator, loss_function, training_strategy):
        """Run one epoch of validation. Returns a dict of validation results."""
        # Initialize data feeding from iterator
        iterator = validation_batch_iterator(data_pool)
        validation_generator = threaded_generator_from_iterator(iterator)

        number_of_batches = len(data_pool) // validation_batch_iterator.batch_size

        validation_mungos_from = []
        validation_mungos_to = []
        validation_predicted_classes = []
        validation_target_classes = []
        validation_results = collections.OrderedDict()
        losses = []

        for current_batch_index in range(number_of_batches):

            # Validation iterator might also output the MuNGOs,
            # for improved evaluation options.
            validation_batch = next(validation_generator)
            mungos_from, mungos_to = None, None
            if len(validation_batch) == 4:
                mungos_from, mungos_to, np_inputs, np_targets = validation_batch
            else:
                np_inputs, np_targets = validation_batch

            inputs = Variable(torch.from_numpy(np_inputs).float())
            targets = Variable(torch.from_numpy(np_targets).float())
            if self.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            predictions = self.net(inputs)
            np_predictions = self.__torch2np(predictions)
            np_predicted_classes = targets2classes(np_predictions)
            np_target_classes = targets2classes(np_targets)

            loss = loss_function(predictions, targets)
            losses.append(self.__torch2np(loss))

            # Compute all evaluation metrics available for current batch.
            current_batch_results = collections.OrderedDict()
            current_batch_results['loss'] = self.__torch2np(loss)
            current_batch_metrics = self.__evaluate_classification(np_predicted_classes, np_target_classes)
            for k, v in current_batch_metrics.items():
                current_batch_results[k] = v

            validation_predicted_classes.extend(np_predicted_classes)
            validation_target_classes.extend(np_target_classes)
            if mungos_from is not None:
                validation_mungos_from.extend(mungos_from)
                validation_mungos_to.extend(mungos_to)

            # Log sample outputs. Used mostly for sanity/debugging.
            _first_n_batch_results_to_print = 3
            if current_batch_index < _first_n_batch_results_to_print:
                logging.info('\t{}: Targets: {}'.format(current_batch_index, np_targets[:10]))
                logging.info('\t{}: Outputs: {}'.format(current_batch_index, np_predictions[:10]))

        # Compute evaluation metrics aggregated over validation set.
        aggregated_metrics = self.__evaluate_classification(validation_predicted_classes,
                                                            validation_target_classes)
        for k, v in aggregated_metrics.items():
            validation_results[k] = v
        aggregated_loss = numpy.mean(losses)
        validation_results['loss'] = aggregated_loss

        # Compute mistakes signatures per class pair
        class_pair_results = eval_clf_by_class_pair(validation_mungos_from,
                                                    validation_mungos_to,
                                                    validation_target_classes,
                                                    validation_predicted_classes)

        class_pair_results['all'] = validation_results
        return class_pair_results

    @staticmethod
    def __evaluate_classification(predicted_classes, true_classes):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision, recall, f_score, true_sum = precision_recall_fscore_support(true_classes,
                                                                               predicted_classes)
        return {'acc': accuracy,
                'prec': precision,
                'rec': recall,
                'fsc': f_score,
                'support': true_sum}

    def __update_learning_rate(self, optimizer, multiplier):
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
            param_group['lr'] = learning_rate * multiplier
            print('\tUpdate learning rate to {0}'.format(param_group['lr']))

    @staticmethod
    def __flatten_validation_results(validation_results):
        """Flattens the validation results structure into a single
        dict. Uses two underscores as separator."""
        output = {}
        for img_name in validation_results:
            r = validation_results[img_name]
            for label in r:
                if label == 'loss':
                    output['{0}/loss'] = r[label]
                    continue
                for k, v in r.items():
                    output['{0}/{1}/{2}'.format(img_name, label, k)] = v

        return output

    @staticmethod
    def __aggregate_validation_results(validation_results):
        losses_per_image = []
        rs_per_label = {}
        for img_name in validation_results:
            for label in validation_results[img_name]:
                if label == 'loss':
                    losses_per_image.append(validation_results[img_name][label])
                    continue
                if label not in rs_per_label:
                    rs_per_label[label] = collections.defaultdict(list)
                for k, v in validation_results[img_name][label].items():
                    rs_per_label[label][k].append(v)

        aggregated_metrics = set(['tp', 'fp', 'fn', 'dice', 'prec', 'rec', 'fsc'])

        agg_results_per_label = {}
        for label in rs_per_label:
            agg_r = {}
            for k, v in rs_per_label[label].items():
                if k not in aggregated_metrics:
                    continue
                agg_r[k] = numpy.mean(v)
            agg_results_per_label[label] = agg_r

        agg_rs = collections.defaultdict(list)
        for label in agg_results_per_label:
            for k, v in agg_results_per_label[label].items():
                if k not in aggregated_metrics:
                    continue
                agg_rs[k].append(v)
        agg_overall = {k: numpy.mean(vs) for k, vs in agg_rs.items()}
        agg_overall['loss'] = numpy.mean(losses_per_image)

        return agg_results_per_label, agg_overall

    def __train_epoch(self, data_pool, training_batch_iterator, loss_function, optimizer,
                      estimate_dices=False,
                      debug_mode=False):
        """Run one epoch of training."""
        colored_command_line = ColoredCommandLine()

        # Initialize data feeding from iterator
        iterator = training_batch_iterator(data_pool)
        generator = threaded_generator_from_iterator(iterator)

        n_batches = len(data_pool) // training_batch_iterator.batch_size

        # Monitors
        batch_train_losses = []

        with tqdm(total=n_batches + 1) as progress_bar:
            # Training loop, one epoch
            for current_batch_index, data_point in enumerate(generator):
                # _data_point = next(generator)
                # print("Batch {} / {}".format(batch_idx, n_batches))
                np_inputs, np_targets = data_point  # next(generator)

                # This assumes that the generator does not already
                # return Torch Variables, which the model can however
                # specify in its prepare() function(s).
                inputs = Variable(torch.from_numpy(np_inputs).float())
                targets = Variable(torch.from_numpy(np_targets).float())
                if self.cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                # One training update:
                optimizer.zero_grad()
                predictions = self.net(inputs)
                loss = loss_function(predictions, targets)
                loss.backward()
                optimizer.step()

                # Monitors update
                batch_train_losses.append(self.__torch2np(loss))

                # Logging during training
                progress_bar.set_description(
                    "Training (current batch loss: {0:.2f})".format(numpy.mean(batch_train_losses)), refresh=False)
                progress_bar.update()

        # Aggregate monitors
        avg_train_loss = numpy.mean(batch_train_losses)

        output = {
            'number': 0,
            'train_loss': avg_train_loss,
            'train_dices': None,
        }

        return output

    def __torch2np(self, var):
        """Converts the PyTorch Variable or tensor to numpy."""
        if self.cuda:
            output = var.data.cpu().numpy()
        else:
            output = var.data.numpy()

        return output

    def __np2torch(self, ndarray):
        output = Variable(torch.from_numpy(ndarray).float())
        if self.cuda:
            output = output.cuda()
        return output
