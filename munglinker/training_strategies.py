from torch.nn.modules.loss import BCELoss
from torch.optim import Adam, Adadelta


class PyTorchTrainingStrategy(object):
    def __init__(self,
                 name=None,
                 initial_learning_rate=0.001,
                 max_epochs=1000,
                 batch_size=2,
                 refine_batch_size=False,
                 improvement_patience=20,
                 optimizer_class=Adam,
                 loss_fn_class=BCELoss,
                 loss_fn_kwargs=dict(),
                 validation_use_detector=False,
                 validation_detector_threshold=0.5,
                 validation_detector_min_area=5,
                 validation_subsample_window=None,
                 validation_stride_ratio=2,
                 validation_no_detector_subsample_window=None,
                 validation_outputs_dump_root=None,
                 best_model_by_fscore=False,
                 n_epochs_per_checkpoint=10,
                 checkpoint_export_file=None,
                 early_stopping=True,
                 lr_refinement_multiplier=0.2,
                 n_refinement_steps=5,
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

        :param validation_no_detector_subsample_window: When validating without
            a detector (such as on non-checkpoint epochs, where we only want
            the validation loss), subsample this window.

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
        self.initial_learning_rate = initial_learning_rate

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
        self.validation_nodetector_subsample_window = validation_no_detector_subsample_window

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

    def loss_function(self):
        return self.loss_fn(**self.loss_fn_kwargs)
