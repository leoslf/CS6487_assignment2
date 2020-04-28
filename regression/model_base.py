import os

from datetime import datetime
from regression.utils import *

class CustomEarlyStopping(EarlyStopping):

    def __init__(self, target=None, **kwargs):
        self.target = target
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        current = self.get_monitor_value(logs)
        if not self.target or self.monitor_op(self.target, self.best):
            super().on_epoch_end(epoch, logs)

class BaseModel:
    def __init__(self,
                 input_shape = (3, 3),
                 output_shape = (3, ),
                 batch_size = None,
                 epochs = 1000,
                 verbose = 2,
                 use_multiprocessing = False,
                 compiled = False,
                 *argv, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.compiled = compiled
        self.epochs = epochs
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.__dict__.update(kwargs)

        self.init()
        self.model = self.prepare_model()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        try:
            self.load_weights()
        except:
            raise ImportError("Could not load pretrained model weights")

        if not self.compiled:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            print ("compiled: %s" % self.__class__.__name__)

        self.model.summary()

    @property
    def name(self):
        return self.__class__.__name__

    def init(self):
        pass

    @property
    def optimizer(self):
        return Adadelta()

    @property
    def loss(self):
        return "mean_squared_error"

    @property
    def weight_filename(self):
        return "%s.h5" % self.name

    def load_weights(self, filename = None):
        if filename is None:
            filename = self.weight_filename

        if os.path.exists(filename):
            self.model.load_weights(filename, by_name=True, skip_mismatch=True)

    def save_weights(self):
        self.model.save_weights(self.weight_filename)

    @property
    def metrics(self):
        return [] # "mean_squared_error"]

    @property
    def use_earlystopping(self):
        return False

    @property
    def earlystopping(self):
       return EarlyStopping(monitor="val_loss", # use validation accuracy for stopping
                            min_delta = 0.0001,
                            patience = 50, 
                            verbose = self.verbose,
                            mode="auto")

    @property
    def modelcheckpoint(self):
        return ModelCheckpoint(os.path.join(self.logdir, "epoch{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), monitor="val_loss", save_weights_only=True, save_best_only=True, period=3)


    @property
    def callbacks(self):
        callbacks = [
            self.modelcheckpoint,
            TensorBoard(log_dir=self.logdir),
            TerminateOnNaN(),
        ]
        if self.use_earlystopping:
            callbacks.append(self.earlystopping)

        return callbacks

    @property
    def logdir(self):
        return "logs/%s/%s" % (self.__class__.__name__, datetime.now().strftime("%Y%m%d-%H%M%S"))

    def prepare_model(self):
        return None

    def fit(self, train_X, train_Y, validation_X, validation_Y):
        history = self.model.fit(train_X, train_Y,
                                 validation_data = (validation_X, validation_Y),
                                 batch_size = self.batch_size,
                                 epochs = self.epochs,
                                 callbacks = self.callbacks,
                                 verbose = self.verbose,
                                 use_multiprocessing = self.use_multiprocessing)
        self.save_weights()
        return history

    def evaluate(self, test_X, test_Y):
        return self.model.evaluate(test_X, test_Y,
                                   batch_size = self.batch_size,
                                   verbose = self.verbose,
                                   use_multiprocessing = self.use_multiprocessing)

    def predict(self, X, *argv, **kwargs):
        return self.model.predict(X, *argv, **kwargs)
