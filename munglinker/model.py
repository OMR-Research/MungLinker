"""This file defines the pytorch fit() wrapper.
"""
from __future__ import print_function, unicode_literals, division

import collections
import logging
import os
import sys
import time

import numpy
import numpy.random
from scipy.misc import imsave, imread


# This one is a really external dependency
from munglinker.batch_iterators import threaded_generator_from_iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss, _assert_no_grad, MSELoss
from torch.optim import Adam

from munglinker.image_normalization import auto_invert, stretch_intensity, ImageNormalizer
from munglinker.utils import BColors, show_batch_simple, show_onset_sequence_predictions

torch.set_default_tensor_type('torch.FloatTensor')

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class FocalLossElemwise(_WeightedLoss):
    """Elementwise Focal Loss implementation for arbitrary tensors that computes
    the loss element-wise, e.g. for semantic segmentation. Alpha-balancing
    is implemented for entire minibatches instead of per-channel.

    See: https://arxiv.org/pdf/1708.02002.pdf  -- RetinaNet
    """
    def __init__(self, weight=None, size_average=True, gamma=0, alpha_balance=False):
        super(FocalLossElemwise, self).__init__(weight, size_average)
        self.gamma = gamma
        self.alpha_balance = alpha_balance

    def forward(self, input, target):
        _assert_no_grad(target)
        # Assumes true is a binary tensor.
        # All operations are elemntwise, unless stated otherwise.
        # ELementwise cross-entropy
        # xent = true * torch.log(pred) + (1 - true) * torch.log(1 - pred)

        # p_t
        p_t = input * target   # p   if y == 1    -- where y is 0, this produces a 0
        p_t += (1 - input) * (1 - target)     # 1 - p   if y == 0      -- where y = 1, this adds 0

        fl_coef = -1 * ((1 - p_t) ** self.gamma)

        # Still elementwise...
        fl = fl_coef * torch.log(p_t)

        if self.alpha_balance:
            alpha = target.sum() / target.numel()
            alpha_t = alpha * input + (1 - alpha) * (1 - input)
            fl *= alpha_t

        # Now we would aggregate this
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


##############################################################################


class PyTorchTrainingStrategy(object):
    def __init__(self,
                 name=None,
                 ini_learning_rate=0.001,
                 max_epochs=1000,
                 batch_size=2,
                 refine_batch_size=False,
                 improvement_patience=50,
                 optimizer_class=Adam,
                 loss_fn_class=MSELoss,   # TODO: Back to softmax? This will be bad...
                 loss_fn_kwargs=dict(),
                 validation_use_detector=False,
                 validation_detector_threshold=0.5,
                 validation_detector_min_area=5,
                 validation_subsample_window=None,
                 validation_stride_ratio=2,
                 validation_nodetector_subsample_window=None,
                 validation_subsample_n=4,
                 validation_outputs_dump_root=None,
                 best_model_by_fscore=False,
                 n_epochs_per_checkpoint=10,
                 checkpoint_export_file=None,
                 early_stopping=True,
                 lr_refinement_multiplier=0.2,
                 n_refinement_steps=5,
                 refinement_patience=50,
                 best_params_file=None,
                 ):
        """Initialize a training strategy. Includes some validation params.

        :param name: Name the training strategy. This is useful for
            keeping track of log files: logging will use this name,
            e.g. as a TensorBoard comment.

        :param ini_learning_rate: Initial learning rate. Passed to
            the optimizer

        :param max_epochs:

        :param batch_size:

        :param refine_batch_size: Someone said "don't decrease learning
            rate, increase batch size". Let's try this strategy somehow.

        :param improvement_patience: How many epochs are we willing to
            train without seeing improvement on the validation data before
            early-stopping?

        :param optimizer_class: A PyTorch optimizer class, like Adam.
            (The *class*, not an *instance* of the class.)

        :param loss_fn_class: A PyTorch loss class, like BCEWithLogitsLoss.
            (The *class*, not an *instance* of the class.)

        :param loss_fn_kwargs: Additional arguments that the loss function
            will need for initialization.

        :param validation_use_detector: Should validation include evaluating
            detector performance? If not, only the validation loss and dice
            coefficient is measured.

        :param validation_detector_threshold: Probability mask threshold.

        :param validation_detector_min_area: Discard small detected regions.

        :param validation_subsample_window: When validating, only compute
            validation metrics from a window of this size randomly subsampled
            from each validation image.

        :param validation_stride_ratio: How much should the detector windows
            overlap? The ratio is the square root of the expected number of
            samples for each pixel (...close enough to the center).

        :param validation_nodetector_subsample_window: When validating without
            a detector (such as on non-checkpoint epochs, where we only want
            the validation loss), subsample this window.

        :param validation_subsample_n: When subsampling the validation window,
            re-sample this many times to get a lower-variance estimate of the
            validation performance.

        :param validation_outputs_dump_root: Dump validation result images
            (prob. masks, prob. maps and predicted labels) into this directory
            plus the ``name`` (so that the dump root can be shared between
            strategies and the images won't mix up).

        :param best_model_by_fscore: Use the validation aggregated f-score
            instead of the loss to keep the best model during training.

        :param early_stopping: Perform early-stopping & refinement?

        :param checkpoint_export_file: Where to dump the checkpoint params?

        :param n_epochs_per_checkpoint: Dump checkpoint params export once per
            this many epochs.

        :param n_refinement_steps: How many times should we try to attenuate
            the learning rate.

        :param lr_refinement_multiplier: Attenuate learning rate by this
            multiplicative factor in each refinement step.

        :param refinement_patience: How many epochs do we wait for improvement
            when in the refining stage.

        :param best_params_file: The file to which to save the best model.
        """
        self.name = name

        # Learning rate
        self.ini_learning_rate = ini_learning_rate

        # Epochs & batches
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.refine_batch_size = refine_batch_size
        self.improvement_patience = improvement_patience

        # Loss function
        self.loss_fn = loss_fn_class
        self.loss_fn_kwargs = loss_fn_kwargs

        # Optimizer
        self.optimizer_class = optimizer_class

        # Validation
        self.validation_use_detector = validation_use_detector
        self.validation_detector_threshold = validation_detector_threshold
        self.validation_detector_min_area = validation_detector_min_area
        self.validation_stride_ratio = validation_stride_ratio
        self.validation_subsample_window = validation_subsample_window
        self.validation_nodetector_subsample_window = validation_nodetector_subsample_window

        self.validation_outputs_dump_root = validation_outputs_dump_root

        # Model selection
        self.best_model_by_fscore = best_model_by_fscore

        # Checkpointing
        self.n_epochs_per_checkpoint = n_epochs_per_checkpoint
        self.checkpoint_export_file = checkpoint_export_file

        # Early-stopping & Refinement
        self.early_stopping = early_stopping
        self.lr_refinement_multiplier = lr_refinement_multiplier
        self.n_refinement_steps = n_refinement_steps
        self.refinement_patience = refinement_patience

        # Persisting the model
        self.best_params_file = best_params_file

    def init_loss_fn(self):
        return self.loss_fn(**self.loss_fn_kwargs)


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
    def __init__(self, net, print_architecture=False):
        self.net = net
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.net.cuda()

        # Logging
        self._tb = None

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
        col = BColors()

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
            self._tb = SummaryWriter(os.path.join(tensorboard_log_path,
                                                  training_strategy.name),
                                     comment=training_strategy.name)

        # Adaptive learning rate..?
        lr = training_strategy.ini_learning_rate

        # We don't need to create iter functions.

        # Initialize evaluation outputs
        tr_results, va_results = [], []

        # Train and validation loss, for early stopping
        tr_losses, va_losses = [], []

        # Early stopping
        last_improvement = 0
        best_model = self.net.state_dict()

        # Refinement
        refinement_stage = False
        refinement_steps = 0

        # Tracking
        best_va_dice = 0.0
        best_tr_loss, best_va_loss = 1e7, 1e7
        prev_fsc_tr, prev_fsc_va = 0.0, 0.0

        print("Strating training...")
        _start_time = time.time()
        try:

            ##################################################################
            # Preparation

            best_loss = best_tr_loss
            #if training_strategy.best_model_by_fscore:
            #    best_loss = prev_fsc_tr

            # Set batch sizes.
            data['train'].batch_size = training_strategy.batch_size

            # Initialize loss and optimizer
            loss_fn = training_strategy.init_loss_fn()
            optimizer = training_strategy.optimizer_class(self.net.parameters(),
                                                          lr=lr)

            # Initialize validation
            data['valid'].batch_size = 1
            val_detector = None
            if training_strategy.validation_use_detector:
                pass
                # # Initialize the detector for evaluating detection
                # # performance, as opposed to mere Dice coef.
                # val_detector = ConnectedComponentDetector(
                #     net=self.net,
                #     is_logits=True,
                #     stride_ratio=training_strategy.validation_stride_ratio,
                #     threshold=training_strategy.validation_detector_threshold,
                #     min_area=training_strategy.validation_detector_min_area,
                # )

            ##################################################################
            # Iteration
            for epoch_idx in range(training_strategy.max_epochs):

                # print('Epoch {0}'.format(epoch_idx))

                ##################################################################
                # Train the epoch
                tr_epoch = self._train_epoch(data['train'],
                                             batch_iters['train'],
                                             loss_fn,
                                             optimizer)
                tr_results.append(tr_epoch)
                tr_loss = tr_epoch['train_loss']
                tr_losses.append(tr_loss)

                # Checkpointing
                if (epoch_idx + 1) % training_strategy.n_epochs_per_checkpoint == 0:
                    if training_strategy.checkpoint_export_file:
                        torch.save(self.net.state_dict(),
                                   training_strategy.checkpoint_export_file)
                    else:
                        logging.warning('Cannot checkpoint: no checkpoint file'
                                        ' specified in training strategy!')

                ##################################################################
                # Validation: run with detector on checkpoints, otherwise
                # only collect loss.
                print('\nValidating epoch {}'.format(epoch_idx))

                # If no early-stopping should be performed, do not bother with validation.
                if not training_strategy.early_stopping:
                    continue

                elif (epoch_idx + 1) % training_strategy.n_epochs_per_checkpoint != 0:
                    ##################################################################
                    # Outside checkpoint.
                    # Only run validation without detector, to track validation loss
                    # for early-stopping & refinement
                    va_epoch = self._validate_epoch(data['valid'],
                                                    batch_iters['valid'],
                                                    loss_fn, training_strategy,
                                                    detector=None,
                                                    subsample_window=training_strategy.validation_nodetector_subsample_window)
                    _, va_epoch_agg = self.__aggregate_validation_results(va_epoch)

                    va_loss = va_epoch_agg['loss']
                    va_losses.append(va_loss)
                    # print('\tValidation loss: {0:.6f}'.format(va_loss))

                else:
                    #########################################################
                    # This only happens once per n_epochs_per_checkpoint
                    va_epoch = self._validate_epoch(data['valid'],
                                                    batch_iters['valid'],
                                                    loss_fn, training_strategy,
                                                    detector=val_detector)
                    va_results.append(va_epoch)
                    va_epoch_agg_l, va_epoch_agg = self.__aggregate_validation_results(va_epoch)

                    self.print_validation_results(va_epoch_agg_l, va_epoch_agg)

                    if self._tb is not None:
                        print('Logging validation epoch results to tensorboard'
                              ' is not implemented!')
                        # self.log_epoch_to_tb(epoch_idx,
                        #                      tr_epoch,
                        #                      va_epoch_agg,
                        #                      va_epoch_agg_l)

                    va_loss = va_epoch_agg['loss']
                    va_losses.append(va_loss)

                ##################################
                # Log validation loss

                epoch_loss = va_losses[-1]
                _va_loss_str = '\tValidation loss: {0:.6f}\t(Patience: {1})' \
                               ''.format(epoch_loss,
                                         training_strategy.improvement_patience - last_improvement)
                if epoch_loss < best_loss:
                    print(col.print_colored(_va_loss_str, col.OKGREEN), end="\n")
                else:
                    print(_va_loss_str)

                ###################################
                # Early-stopping: Check for improvement
                if epoch_loss < best_loss:
                    # If the f-score is used, it is inverted in both epoch_loss
                    # and best_loss.
                    last_improvement = best_loss - epoch_loss
                    best_loss = epoch_loss
                    best_model = self.net.state_dict()
                    last_improvement = 0
                else:
                    last_improvement += 1

                # Early-stopping: continue, refine, or end training
                if (refinement_stage and (last_improvement > training_strategy.refinement_patience)) \
                    or (last_improvement > training_strategy.improvement_patience):
                    print('Early-stopping: exceeded patience in epoch {0}'
                          ''.format(epoch_idx))

                    last_improvement = 0
                    refinement_stage = True
                    if refinement_steps < training_strategy.n_refinement_steps:
                        self.update_learning_rate(optimizer,
                                                  training_strategy.lr_refinement_multiplier)
                        refinement_steps += 1

                    else:
                        print('Early-stopping: exceeded refinement budget,'
                              ' training ends.')

                        print('---------------------------------')
                        print('Final validation:\n')

                        va_epoch = self._validate_epoch(data['valid'], loss_fn, training_strategy,
                                                        val_detector)
                        va_results.append(va_epoch)

                        # (Aggregate across images, macro-averages per label and overall)
                        va_epoch_agg_l, va_epoch_agg = self.__aggregate_validation_results(va_epoch)

                        self.print_validation_results(va_epoch_agg_l, va_epoch_agg)

                        # Log results to TensorBoard
                        if self._tb is not None:
                            # Log epoch results to tensorboard.
                            # Training results:
                            self.log_epoch_to_tb(epoch_idx,
                                                 tr_epoch,
                                                 va_epoch_agg,
                                                 va_epoch_agg_l)

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

    def log_epoch_to_tb(self, epoch_idx, tr_epoch, va_epoch_agg, va_epoch_agg_l):
        self._tb.add_scalar('train/loss',
                            tr_epoch['train_loss'], epoch_idx)
        tr_d = tr_epoch['train_dices']
        if tr_d is not None:
            for thr in tr_d:
                self._tb.add_scalar('train/dice_th{0}'.format(thr),
                                    tr_d[thr], epoch_idx)

        # Validation results:
        # for k, v in self.__flatten_validation_results(va_epoch).items():
        #     self._tb.add_scalar('val/{0}'.format(k), v, epoch_idx)
        for label in va_epoch_agg_l:
            for k, v in va_epoch_agg_l[label].items():
                self._tb.add_scalar('{0}/{1}'.format(label, k),
                                    v, epoch_idx)
        for k, v in va_epoch_agg.items():
            self._tb.add_scalar('{0}'.format(k, v, epoch_idx),
                                v, epoch_idx)

    def print_validation_results(self, val_label_avg_results, val_avg_results):
        """Prints the results for each label, macro-averaged over images,
        and macro-averaged over each label.

        :param val_label_avg_results: Dict of results per label. Each dict
            element is expected to be again a dict with at least ``dice``,
            ``prec``, ``rec`` and ``fsc`` element.

        :param val_avg_results: A dict with the ``dice``, ``prec``, ``rec``
            and ``fsc`` keys.
        """
        for output_label in val_label_avg_results:
            # print('\t{0}:'.format(output_label))
            l_val_results = val_label_avg_results[output_label]
            print('{4}:\t\tDice: {0:.3f},'
                  ' Precision: {1:.3f},'
                  ' Recall: {2:.3f},'
                  ' F-score: {3:.3f}'
                  ''.format(l_val_results['dice'],
                            l_val_results['prec'],
                            l_val_results['rec'],
                            l_val_results['fsc'],
                            output_label))

        if len(val_label_avg_results) > 1:
            print('All:\t\tDice: {0:.3f},'
                  ' Precision: {1:.3f},'
                  ' Recall: {2:.3f},'
                  ' F-score: {3:.3f}'
                  ''.format(val_avg_results['dice'],
                            val_avg_results['prec'],
                            val_avg_results['rec'],
                            val_avg_results['fsc']))

    def _validate_epoch(self, data_pool, valid_batch_iter,
                        loss_fn, training_strategy,
                        detector=None,
                        subsample_window=None,
                        estimate_dices=False,
                        debug_mode=False):
        """Run one epoch of validation. Returns a dict of validation results
        per validation image and (nested) output label."""
        # print('Validating model...')

        # This is what we eventually want, but let's do this after we have
        # debugged trainings:
        #
        # if detector is not None:
        #     return evaluate_detection(data_loader=valid_loader, detector=detector,
        #                               show_results=False,
        #                               subsample_window=None)
        # Initialize data feeding from iterator
        iterator = valid_batch_iter(data_pool)
        val_generator = threaded_generator_from_iterator(iterator)

        n_batches = len(data_pool) // valid_batch_iter.batch_size

        validation_results = collections.OrderedDict()
        validation_preds = collections.OrderedDict()
        validation_targets = collections.OrderedDict()
        #
        # validation_origs = collections.OrderedDict()
        # validation_prob_maps = collections.OrderedDict()
        # validation_prob_masks = collections.OrderedDict()
        # validation_pred_labels = collections.OrderedDict()

        for batch_idx in range(n_batches):
            # if detector is not None:
            #     print('\tImage {0}: {1}'.format(i, val_img_name))

            np_inputs, np_targets = next(val_generator)

            inputs = Variable(torch.from_numpy(np_inputs).float())
            targets = Variable(torch.from_numpy(np_targets).float())
            if self.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            pred = self.net(inputs)

            loss = loss_fn(pred, targets)

            if batch_idx < 5:
                # This will get relegated to logging; for now
                # we print directly.
                np_pred = pred.data.numpy()
                np_pred_int = (np_pred - (np_pred % 1.0)).astype(numpy.int)
                # print('\t{}: Targets: {}'.format(batch_idx, np_targets))
                # print('\t{}: Errors:  {}'.format(batch_idx, np_pred_int - np_targets))

            if (batch_idx < 1) and (self._torch2np(loss) < 100):
                # show_batch_simple(np_inputs, np_targets)
                # show_onset_counter_predictions(inputs, targets, self.net)
                # show_onset_sequence_predictions(inputs, targets, self.net)
                pass

            current_batch_results = collections.OrderedDict()
            current_batch_results['loss'] = self._torch2np(loss)

            validation_results[batch_idx] = current_batch_results

        # # Dumping output images -- only with detector
        # if detector is not None:
        #     if training_strategy.validation_outputs_dump_root is not None:
        #         if self.net.n_output_channels > 1:
        #             print('(Cannot currently dump results for more than 1 output channel.)')
        #         else:
        #             output_dir = os.path.join(training_strategy.validation_outputs_dump_root,
        #                                       training_strategy.name)
        #             if not os.path.isdir(output_dir):
        #                 os.mkdir(output_dir)
        #             self.dump_validation_results(output_dir,
        #                                          validation_origs,
        #                                          validation_prob_maps,
        #                                          validation_prob_masks,
        #                                          validation_pred_labels)

        return validation_results

    def update_learning_rate(self, optimizer, multiplier):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            param_group['lr'] = lr * multiplier
            print('\tUpdate learning rate to {0}'.format(param_group['lr']))

    @staticmethod
    def dump_validation_results(dump_root, origs, maps, masks, labels):
        """Dumps the outputs of the detector into the given root."""
        origs_dir = os.path.join(dump_root, 'orig')
        if not os.path.isdir(origs_dir):
            os.mkdir(origs_dir)
        for val_img_name in origs:
            filename = os.path.join(origs_dir, val_img_name + '.png')
            img = origs[val_img_name]
            # img *= (255 / img.max())
            imsave(filename, img)


        masks_dir = os.path.join(dump_root, 'prob_mask')
        if not os.path.isdir(masks_dir):
            os.mkdir(masks_dir)
        for val_img_name in masks:
            filename = os.path.join(masks_dir, val_img_name + '.png')
            img = masks[val_img_name] * 255
            imsave(filename, img)

        maps_dir = os.path.join(dump_root, 'prob_map')
        if not os.path.isdir(maps_dir):
            os.mkdir(maps_dir)
        for val_img_name in maps:
            filename = os.path.join(maps_dir, val_img_name + '.png')
            img = maps[val_img_name] * 255
            imsave(filename, img)

        labels_dir = os.path.join(dump_root, 'pred_labels')
        if not os.path.isdir(labels_dir):
            os.mkdir(labels_dir)
        for val_img_name in labels:
            filename = os.path.join(labels_dir, val_img_name + '.png')
            img = labels[val_img_name]
            imsave(filename, img)

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

    def _train_epoch(self, data_pool, train_batch_iter, loss_fn, optimizer,
                     estimate_dices=False,
                     debug_mode=False):
        """Run one epoch of training."""
        col = BColors()

        # Initialize data feeding from iterator
        iterator = train_batch_iter(data_pool)
        generator = threaded_generator_from_iterator(iterator)

        n_batches = len(data_pool) // train_batch_iter.batch_size
        # print('n. of train entities: {}'.format(len(data_pool)))
        # print('batch size: {}'.format(train_batch_iter.batch_size))

        # Monitors
        batch_train_losses = []
        dice_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        train_dices_per_threshold = {thr: [] for thr in dice_thresholds}

        # Time tracking
        _batch_times = numpy.zeros(5, dtype=numpy.float32)
        _epoch_start = time.time()
        _after = time.time()

        # Training loop, one epoch
        for batch_idx, _data_point in enumerate(generator):
            # _data_point = next(generator)
            # print("Batch {} / {}".format(batch_idx, n_batches))
            np_inputs, np_targets = _data_point  # next(generator)

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
            preds = self.net(inputs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            # Monitors update
            batch_train_losses.append(self._torch2np(loss))
            if estimate_dices:
                from mhr.experiments.fcn.evaluation import dice
                np_preds = self._torch2np(preds)
                for thr in dice_thresholds:
                    for i in range(np_inputs.shape[0]):
                        seg = np_preds[i] > thr
                        train_dices_per_threshold[thr].append(dice(seg,
                                                                   np_targets[i]))

            # Train time tracking update
            _batch_time = time.time() - _after
            _after = time.time()
            _train_time = _after - _epoch_start

            _batch_times[0:4] = _batch_times[1:5]
            _batch_times[4] = _batch_time
            _ups = 1.0 / _batch_times.mean()

            # Logging during training
            perc = 100 * (float(batch_idx) / n_batches)
            dec = int(perc // 4)
            progbar = "|" + dec * "#" + (25 - dec) * "-" + "|"
            vals = (perc, progbar, _train_time, _ups,
                    numpy.mean(batch_train_losses))
            loss_str = " (%d%%) %s time: %.2fs, ups: %.2f, loss: %.5f" % vals
            if batch_idx != n_batches - 1:
                print(col.print_colored(loss_str, col.WARNING), end="\r")
            else:
                print(col.print_colored(loss_str, col.OKBLUE), end="\n")
            sys.stdout.flush()

            # Visualizing performance on training data
            if ((batch_idx % 100) == 0) and (self._torch2np(loss) < 100):
                pass
                # show_batch_simple(np_inputs, np_targets)
                # show_onset_counter_predictions(inputs, targets, self.net)
                # show_onset_sequence_predictions(inputs, targets, self.net)



        # Aggregate monitors
        avg_train_loss = numpy.mean(batch_train_losses)
        if estimate_dices:
            train_dices = {thr: numpy.mean(train_dices_per_threshold[thr])
                           for thr in dice_thresholds}

        output = {
            'number': 0,
            'train_loss': avg_train_loss,
            'train_dices': None,
        }
        if estimate_dices:
            output['train_dices'] = train_dices

        return output

    def _torch2np(self, var):
        """Converts the PyTorch Variable or tensor to numpy."""
        if self.cuda:
            output = var.data.cpu().numpy()
        else:
            output = var.data.numpy()

        return output

    def _np2torch(self, ndarray):
        output = Variable(torch.from_numpy(ndarray).float())
        if self.cuda:
            output = output.cuda()
        return output
