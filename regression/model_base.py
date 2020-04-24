import os

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow as tf

    from keras.models import *
    from keras.layers import *
    from keras.initializers import *
    from keras.optimizers import *
    from keras.objectives import *
    from keras.callbacks import * 
    from keras.losses import * 

    from keras import backend as K
    from keras.utils import generic_utils

from datetime import datetime
from regression.utils import *

class BaseModel:
    def __init__(self,
                 input_shape = (9, ),
                 output_shape = (3, ),
                 batch_size = None,
                 epochs = 1000,
                 verbose = 1,
                 use_multiprocessing = False,
                 *argv, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.__dict__.update(kwargs)

        self.init()
        self.model = self.prepare_model()
        self.model.compile(loss=self.loss, optimizer="adam", metrics=self.metrics)

    def init(self):
        pass

    @property
    def loss(self):
        return "mean_squared_error"

    @property
    def metrics(self):
        return ["mean_squared_error"]

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
        return [
            self.earlystopping,
            self.modelcheckpoint,
            TensorBoard(log_dir=self.logdir,
                        update_freq="epoch",
                        write_graph=True,
                        write_images=False,),
        ]

    @property
    def logdir(self):
        return "logs/%s/%s" % (self.__class__.__name__, datetime.now().strftime("%Y%m%d-%H%M%S"))

    def prepare_model(self):
        return None

    def fit(self, train_X, train_Y, validation_X, validation_Y):
        self.model.fit(train_X, train_Y,
                       validation_data = (validation_X, validation_Y),
                       batch_size = self.batch_size,
                       epochs = self.epochs,
                       callbacks = self.callbacks,
                       verbose = self.verbose,
                       use_multiprocessing = self.use_multiprocessing)

    def evaluate(self, test_X, test_Y):
        return self.model.evaluate(test_X, test_Y,
                                   batch_size = self.batch_size,
                                   verbose = self.verbose,
                                   use_multiprocessing = self.use_multiprocessing)

    def predict(self, X):
        raise NotImplementedError
