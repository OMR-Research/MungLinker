import collections
import copy
import logging
import os
import pprint
from math import ceil
from typing import Dict, List, Tuple

import numpy
import numpy.random
import torch
from muscima.cropobject import CropObject
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import Module
from tqdm import tqdm

from munglinker.batch_iterators import PoolIterator
from munglinker.data_pool import PairwiseMungoDataPool
from munglinker.evaluation import evaluate_classification_by_class_pairs, print_class_pair_results
from munglinker.training_strategies import PyTorchTrainingStrategy
from munglinker.utils import targets2classes

torch.set_default_tensor_type('torch.FloatTensor')


class PyTorchNetwork(object):
    """The PyTorchNetwork class wraps a PyTorch ``nn.Module`` instance
    with a sklearn-like interface (fit, predict, predict_proba, etc.)
    It should be universal: independent of what net it is wrapping, and
    what task this net is used for; therefore, you have to make sure that
    you are feeding the correct data to the correct model, and that the
    training strategy is correct.

    The biggest workhorse is the ``fit`` function.
    """

    def __init__(self, net: Module, training_strategy: PyTorchTrainingStrategy = None, tensorboard_log_path=None):
        """
        :param net:

        :param training_strategy: A ``NamedTuple``-like class that aggregates
            parameters of the training process: optimizer type, initial LR and
            decay, loss function, etc.

        :param tensorboard_log_path: TensorBoard writer will write to this
            directory.
        """
        self.net = net
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.net.cuda()

        if training_strategy is None:
            self.training_strategy = PyTorchTrainingStrategy()
        else:
            self.training_strategy = training_strategy

        self.tensorboard = None  # type: SummaryWriter
        if tensorboard_log_path is not None:
            os.makedirs(tensorboard_log_path, exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_path, self.training_strategy.name),
                                             comment=self.training_strategy.name)

        # Extra variable for learning rate, since it attenuates during training
        learning_rate = self.training_strategy.initial_learning_rate
        self.optimizer = self.training_strategy.optimizer_class(self.net.parameters(), lr=learning_rate)

        if self.training_strategy.best_params_file is None:
            raise Exception('No file to save the best model is specified in training strategy!')

    def fit(self,
            data,
            batch_iters: Dict[str, PoolIterator],
            dump_file=None,
            log_file=None,
            initial_epoch=1,
            previously_best_validation_loss=numpy.inf):
        """Trains the model.

        :param data: A dict that has a ``'train'`` and a ``'valid'`` key,
            and its values are objects that implement the ``DataPool``
            interface for ``__getitem__``.

        :param batch_iters: A dict that has a ``train`` and ``valid`` key,
            and its values are objects

        :param dump_file: The trained net's state dict will be dumped
            into this file. The dump happens every ecpoch in which the model
            improves, so that the best trained model is always saved.

        :param log_file: Training progress will be logged to this file.

        :param initial_epoch: If resuming the training, provide this parameter

        :param previously_best_validation_loss: If resuming the training, provide the validation loss that was persisted

        """

        if dump_file is not None:
            out_path = os.path.dirname(dump_file)
            if out_path and not os.path.isdir(out_path):
                os.mkdir(out_path)

        if log_file is not None:
            out_path = os.path.dirname(log_file)
            if out_path and not os.path.isdir(out_path):
                os.mkdir(out_path)

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
        best_training_loss, best_validation_loss = numpy.inf, previously_best_validation_loss
        previous_fscore_training, previous_fscore_validation = 0.0, 0.0

        try:

            ##################################################################
            # Preparation
            if self.training_strategy.best_model_by_fscore:
                best_validation_loss = previous_fscore_training

            # Set batch sizes.
            data['train'].batch_size = self.training_strategy.batch_size

            # Initialize loss
            loss_function = self.training_strategy.loss_function

            ##################################################################
            # Iteration
            for current_epoch_index in range(initial_epoch, self.training_strategy.max_epochs + 1):

                ##################################################################
                # Train the epoch
                training_epoch_output = self.__train_epoch(data['train'], batch_iters['train'], loss_function,
                                                           self.optimizer, current_epoch_index)
                training_results.append(training_epoch_output)
                training_loss = training_epoch_output['train_loss']
                training_losses.append(training_loss)

                ##############################################################
                # Checkpointing
                if current_epoch_index % self.training_strategy.n_epochs_per_checkpoint == 0:
                    if self.training_strategy.checkpoint_export_file:
                        if self.training_strategy.checkpoint_export_file is None:
                            os.makedirs("models", exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(self.training_strategy.checkpoint_export_file),
                                        exist_ok=True)
                        torch.save({
                            "epoch": current_epoch_index,
                            "model_state_dict": self.net.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_validation_loss": best_validation_loss},
                            self.training_strategy.checkpoint_export_file)
                    else:
                        logging.warning('Cannot checkpoint: no checkpoint file'
                                        ' specified in training strategy!')

                ##################################################################
                # Validation: run with detector on checkpoints, otherwise
                # only collect loss.
                #########################################################
                # This only happens once per n_epochs_per_checkpoint
                if current_epoch_index % self.training_strategy.n_epochs_per_checkpoint == 0:
                    validation_epoch_output = self.__validate_epoch(data['valid'], batch_iters['valid'], loss_function,
                                                                    current_epoch_index)
                    validation_results.append(validation_epoch_output)

                    self.__log_epoch_to_tensorboard(current_epoch_index,
                                                    training_epoch_output,
                                                    validation_epoch_output)

                    if 'f-score' in validation_epoch_output:
                        validation_loss = -1 * validation_epoch_output['all']['f-score'][-1]
                    else:
                        validation_loss = validation_epoch_output['all']['loss']
                    validation_losses.append(validation_loss)

                    print('Validation results: {}'.format(pprint.pformat(validation_epoch_output['all'])))

                    if self.tensorboard is None:
                        print_class_pair_results(validation_epoch_output)

                ##################################
                # Log validation loss

                epoch_loss = validation_losses[-1]
                if epoch_loss < best_validation_loss:
                    print('Validation loss improved from {0:.3f} to {1:.3f}'.format(best_validation_loss, epoch_loss))
                else:
                    print('Validation loss did not improve over previous best {0:.3f}. Remaining patience: {1}'.format(
                        best_validation_loss,
                        self.training_strategy.improvement_patience - epochs_since_last_improvement))

                ##############################################################
                # Early-stopping: Check for improvement and save best model
                if epoch_loss < best_validation_loss:
                    best_validation_loss = epoch_loss
                    best_model = self.net.state_dict()
                    print('Saving best model: {}'.format(self.training_strategy.best_params_file))
                    torch.save({
                        "epoch": current_epoch_index,
                        "model_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_validation_loss": best_validation_loss},
                        self.training_strategy.best_params_file)

                    epochs_since_last_improvement = 0
                else:
                    epochs_since_last_improvement += 1

                # Early-stopping: continue, refine, or end training
                if (refinement_stage and (epochs_since_last_improvement > self.training_strategy.refinement_patience)) \
                        or (epochs_since_last_improvement > self.training_strategy.improvement_patience):
                    print('Early-stopping: exceeded patience in epoch {0}'
                          ''.format(current_epoch_index + 1))

                    epochs_since_last_improvement = 0
                    refinement_stage = True
                    if refinement_steps < self.training_strategy.number_of_refinement_steps:
                        self.__update_learning_rate(self.optimizer,
                                                    self.training_strategy.lr_refinement_multiplier)
                        refinement_steps += 1

                    else:
                        print('Early-stopping: exceeded refinement budget, training ends.')
                        print('---------------------------------')
                        print('Final validation:\n')

                        validation_epoch_output = self.__validate_epoch(data['valid'], batch_iters['valid'],
                                                                        loss_function,
                                                                        current_epoch_index)
                        validation_results.append(validation_epoch_output)

                        self.__log_epoch_to_tensorboard(current_epoch_index,
                                                        training_epoch_output,
                                                        validation_epoch_output)

                        break

        except KeyboardInterrupt:
            pass

        # Set net to best weights
        self.net.load_state_dict(best_model)

        # Return best validation loss
        if self.training_strategy.best_model_by_fscore:
            return best_validation_loss * -1
        else:
            return best_validation_loss

    def __train_epoch(self, data_pool: PairwiseMungoDataPool, training_batch_iterator: PoolIterator, loss_function,
                      optimizer,
                      current_epoch_index: int):
        """Run one epoch of training."""

        # Initialize data feeding from iterator
        iterator = training_batch_iterator(data_pool)

        number_of_batches = ceil(len(data_pool) / training_batch_iterator.batch_size)

        # Monitors
        batch_train_losses = []
        average_training_loss = numpy.inf
        progress_bar = None

        # Training loop, one epoch
        for current_batch_index, data_batch in enumerate(iterator):

            if progress_bar is None:
                # We create the progress-bar in the first iteration, after iterator-pool has initialized to prevent
                # multiple nested tqdm-progress bars, which do not work properly. One tqdm-progress-bar will be
                # triggered inside the iterator, when it first starts
                progress_bar = tqdm(total=number_of_batches,
                                    desc="Training epoch {0} (average loss: ?)".format(current_epoch_index))

            np_inputs = data_batch["patches"]  # type: numpy.ndarray
            np_targets = data_batch["targets"]  # type: numpy.ndarray

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
            predictions = self.net(inputs).flatten()
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

            # Monitors update
            batch_train_losses.append(self.__torch2np(loss))

            # Aggregate monitors
            average_training_loss = numpy.mean(batch_train_losses)

            # Logging during training
            if progress_bar is not None:
                progress_bar.set_description(
                    "Training epoch {0} (average loss: {1:.3f})".format(current_epoch_index, average_training_loss),
                    refresh=False)
                progress_bar.update()

        if progress_bar is not None:
            progress_bar.close()

        output = {
            'number': 0,
            'train_loss': average_training_loss,
            'train_dices': None,
        }

        return output

    def __validate_epoch(self, data_pool: PairwiseMungoDataPool, validation_batch_iterator, loss_function, current_epoch_index: int):
        """Run one epoch of validation. Returns a dict of validation results."""
        # Initialize data feeding from iterator
        iterator = validation_batch_iterator(data_pool)
        number_of_batches = ceil(len(data_pool) / validation_batch_iterator.batch_size)

        validation_mungos_from = []
        validation_mungos_to = []
        validation_predicted_classes = []
        validation_target_classes = []
        validation_results = collections.OrderedDict()
        losses = []

        for current_batch_index, data_batch in enumerate(tqdm(iterator, total=number_of_batches,
                                                              desc="Validating epoch {0}".format(current_epoch_index))):
            mungos_from = data_batch["mungos_from"]  # type: List[CropObject]
            mungos_to = data_batch["mungo_to"]  # type: List[CropObject]
            np_inputs = data_batch["patches"]  # type: numpy.ndarray
            np_targets = data_batch["targets"]  # type: numpy.ndarray

            inputs = self.__np2torch(np_inputs)
            targets = self.__np2torch(np_targets)

            predictions = self.net(inputs).flatten()
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

        for mung in data_pool.mungs:
            # One mung is one file
            reference_crop_objects = mung.cropobjects # type: List[CropObject]
            doc = reference_crop_objects[0].doc
            inference_crop_objects = [copy.deepcopy(m) for m in mung.cropobjects]
            for m in inference_crop_objects:
                m.outlinks = []
                m.inlinks = []

            id_to_crop_object_mapping = {c.objid: c for c in inference_crop_objects}
            indices = [i for i, m in enumerate(validation_mungos_from) if m.doc == doc]
            mungos_from_this_file = [validation_mungos_from[i] for i in indices]
            mungos_to_this_file = [validation_mungos_to[i] for i in indices]
            output_classes = [validation_predicted_classes[i] for i in indices]
            from munglinker.run import MunglinkerRunner
            for mungo_from, mungo_to, output_class in zip(mungos_from_this_file, mungos_to_this_file, output_classes):
                has_edge = output_class == 1
                if has_edge:
                    MunglinkerRunner.add_edge_in_graph(mungo_from.objid, mungo_to.objid, id_to_crop_object_mapping)

            from munglinker.evaluate_notation_assembly_from_mung import compute_statistics_on_crop_objects
            precision, recall, f1_score, true_positives, false_positives, false_negatives = \
                compute_statistics_on_crop_objects(reference_crop_objects, inference_crop_objects)
            print("Statistics for " + doc)
            print('Precision: {0:.3f}, Recall: {1:.3f}, F1-Score: {2:.3f}'.format(precision, recall, f1_score))
            print("True positives: {0}, False positives: {1}, False Negatives: {2}".format(true_positives, false_positives,
                                                                               false_negatives))

        # Compute evaluation metrics aggregated over validation set.
        aggregated_metrics = self.__evaluate_classification(validation_predicted_classes,
                                                            validation_target_classes)
        for k, v in aggregated_metrics.items():
            validation_results[k] = v
        aggregated_loss = numpy.mean(losses)
        validation_results['loss'] = aggregated_loss

        # Compute mistakes signatures per class pair
        class_pair_results = evaluate_classification_by_class_pairs(validation_mungos_from,
                                                                    validation_mungos_to,
                                                                    validation_target_classes,
                                                                    validation_predicted_classes)

        class_pair_results['all'] = validation_results
        return class_pair_results

    def predict(self, data_pool: PairwiseMungoDataPool, runtime_batch_iterator) -> Tuple[
        List[CropObject], List[CropObject], List[int]]:
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
        iterator = runtime_batch_iterator(data_pool)
        number_of_batches = ceil(len(data_pool) / runtime_batch_iterator.batch_size)
        print('{} runtime entities found. Processing them in {} batches.'.format(len(data_pool), number_of_batches))

        all_mungos_from = []
        all_mungos_to = []
        all_np_predicted_classes = []

        for current_batch_index, data_batch in enumerate(tqdm(iterator, total=number_of_batches,
                                                              desc="Predicting connections")):
            mungos_from = data_batch["mungos_from"]  # type: List[CropObject]
            mungos_to = data_batch["mungo_to"]  # type: List[CropObject]
            np_inputs = data_batch["patches"]  # type: numpy.ndarray

            all_mungos_from.extend(mungos_from)
            all_mungos_to.extend(mungos_to)

            inputs = self.__np2torch(np_inputs)
            predictions = self.net(inputs).flatten()
            np_predictions = self.__torch2np(predictions)
            np_predicted_classes = targets2classes(np_predictions)
            all_np_predicted_classes.extend(np_predicted_classes)

        return all_mungos_from, all_mungos_to, all_np_predicted_classes

    @staticmethod
    def __evaluate_classification(predicted_classes, true_classes):
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision, recall, f_score, true_sum = precision_recall_fscore_support(true_classes, predicted_classes)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f-score': f_score, 'support': true_sum}

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

        aggregated_metrics = {'tp', 'fp', 'fn', 'dice', 'precision', 'recall', 'f-score'}

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
                for key, value in validation_epoch_outputs[label].items():
                    label_name = 'pairwise_{0}/{1}__{2}'.format(key, label[0], label[1])
                    self.tensorboard.add_scalar(label_name, value, epoch_index)

        for key, value in validation_epoch_outputs['all'].items():
            if key in ['accuracy', 'loss']:  # Accuracy and Loss have only one value
                self.tensorboard.add_scalar('validation/{0}'.format(key), value, epoch_index)
            else:  # Precision, Recall and F-Score have two values, for two classes: Negative Samples / Positive Samples
                value_for_negative_class, value_for_positive_class = value[0], value[1]
                self.tensorboard.add_scalar('validation/{0}'.format(key), value_for_positive_class, epoch_index)
