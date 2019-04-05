from torch.nn.modules.loss import BCELoss
from torch.optim import Adam, Adadelta


class PyTorchTrainingStrategy(object):
    def __init__(self,
                 name=None,
                 initial_learning_rate=0.001,
                 max_epochs=1000,
                 batch_size=2,
                 improvement_patience=20,
                 optimizer_class=Adam,
                 loss_fn_class=BCELoss,
                 loss_fn_kwargs=dict(),
                 best_model_by_fscore=False,
                 n_epochs_per_checkpoint=10,
                 checkpoint_export_file=None,
                 lr_refinement_multiplier=0.2,
                 number_of_refinement_steps=5,
                 refinement_patience=5,
                 best_params_file="default_model.tsd",
                 ):
        """Initialize a training strategy. Includes some validation params.

        :param name: Name the training strategy. This is useful for
            keeping track of log files: logging will use this name,
            e.g. as a TensorBoard comment.

        :param initial_learning_rate: Initial learning rate. Passed to
            the optimizer

        :param max_epochs:

        :param batch_size:

        :param improvement_patience: How many epochs are we willing to
            train without seeing improvement on the validation data before
            early-stopping?

        :param optimizer_class: A PyTorch optimizer class, like Adam.
            (The *class*, not an *instance* of the class.)

        :param loss_fn_class: A PyTorch loss class, like BCEWithLogitsLoss.
            (The *class*, not an *instance* of the class.)

        :param loss_fn_kwargs: Additional arguments that the loss function
            will need for initialization.

        :param best_model_by_fscore: Use the validation aggregated f-score
            instead of the loss to keep the best model during training.

        :param checkpoint_export_file: Where to dump the checkpoint params?

        :param n_epochs_per_checkpoint: Dump checkpoint params export once per
            this many epochs.

        :param number_of_refinement_steps: How many times should we try to attenuate
            the learning rate.

        :param lr_refinement_multiplier: Attenuate learning rate by this
            multiplicative factor in each refinement step.

        :param refinement_patience: How many epochs do we wait for improvement
            when in the refining stage.

        :param best_params_file: The file to which to save the best model.
        """
        self.name = name

        # Learning rate
        self.initial_learning_rate = initial_learning_rate

        # Epochs & batches
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.improvement_patience = improvement_patience

        # Loss function
        self.loss_fn = loss_fn_class
        self.loss_fn_kwargs = loss_fn_kwargs

        # Optimizer
        self.optimizer_class = optimizer_class

        # Model selection
        self.best_model_by_fscore = best_model_by_fscore

        # Checkpointing
        self.n_epochs_per_checkpoint = n_epochs_per_checkpoint
        self.checkpoint_export_file = checkpoint_export_file

        # Early-stopping & Refinement
        self.lr_refinement_multiplier = lr_refinement_multiplier
        self.number_of_refinement_steps = number_of_refinement_steps
        self.refinement_patience = refinement_patience

        # Persisting the model
        self.best_params_file = best_params_file

    def loss_function(self):
        return self.loss_fn(**self.loss_fn_kwargs)

    def summary(self) -> str:
        summary_string = "Training Strategy {0}:\n".format(self.name)
        summary_string += "  Training for {0} epochs with a batch-size of {1}.\n".format(self.max_epochs,
                                                                                         self.batch_size)
        summary_string += "  Optimizing with {0}, starting at a Learning rate of {1}\n".format(self.optimizer_class,
                                                                                               self.initial_learning_rate)
        summary_string += "  Loss is computed by {0}, with additional parameters {1}\n".format(self.loss_fn,
                                                                                               self.loss_fn_kwargs)
        summary_string += "  Early stopping after {0} epochs without improvement.\n".format(self.improvement_patience)
        summary_string += "  Checkpointing every {0} epochs into {1}.\n".format(self.n_epochs_per_checkpoint,
                                                                                self.checkpoint_export_file)
        summary_string += "  Saving best model into {0}.\n".format(self.best_params_file)
        if self.best_model_by_fscore:
            summary_string += "  Validating by best F1-Score.\n"
        else:
            summary_string += "  Validating by best validation loss.\n"
        summary_string += "  After early stopping, refining for max {0} epochs with patience of {1} epochs without " \
                          "improvement and a learning rate reduction factor of {2}.\n".format(
            self.number_of_refinement_steps, self.refinement_patience, self.lr_refinement_multiplier)

        return summary_string
