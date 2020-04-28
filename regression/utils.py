import os
import numpy as np
import logging

import functools

from itertools import *
from datetime import datetime

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow as tf

    from keras.models import *
    from keras.layers import *
    from keras.initializers import *
    from keras.optimizers import *
    from keras.regularizers import *
    from keras.objectives import *
    from keras.callbacks import * 
    from keras.losses import * 

    from keras import backend as K
    from keras.utils import generic_utils


logger = logging.getLogger(__name__)

HOURS_PER_DAY = 24.
MINUTES_PER_HOUR = 60.
SECONDS_PER_MINUTE = 60.
MILLISECONDS_PER_SECOND = 1000.
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
SECONDS_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_DAY # 60 * 1,440 = 86,400 SI seconds
MILLISECONDS_PER_DAY = MILLISECONDS_PER_SECOND * SECONDS_PER_DAY # 1000 * 86400 = 86,400,000

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def jd_time(hour, minute, second, microsecond, **kwargs):
    return hour / HOURS_PER_DAY + minute / MINUTES_PER_DAY + second / SECONDS_PER_DAY + microsecond / MILLISECONDS_PER_DAY

def to_dict(object):
    return { attr : getattr(object, attr) for attr in dir(object) }

def normalized_time(time):
    """ Fraction Part of Julian Day Number (ignoring milliseconds)

    Output range: [0, 1]
    """
    return jd_time(**to_dict(time))

def parse_timestamp(timestamp, format="%H:%M:%S"):
    return datetime.strptime(str(timestamp, "utf-8"), format)

def load_csv(filename):
    # uid, {(x_i, y_i, t_i)}_{i = 1}^{4}
    timestamp_columns = 3 * (np.arange(4) + 1)
    timestamp_parser = compose(normalized_time, parse_timestamp)
    return np.loadtxt(filename,
                      delimiter=",",
                      converters = dict(zip(timestamp_columns, repeat(timestamp_parser))))

def drop_userid(dataset):
    """ Dropping the first column """
    return dataset[:, 1:]

def split_XY(dataset, num_features=3, X_records=3, Y_records=1):
    assert dataset.shape[1] == num_features * (X_records + Y_records)
    X = dataset[:, :X_records * num_features].reshape(-1, X_records, num_features)
    Y = dataset[:, X_records * num_features:]
    return X, Y

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            tf.summary.scalar(name, value, epoch)
            # summary = tf.Summary()
            # summary_value = summary.value.add()
            # summary_value.simple_value = value.item()
            # summary_value.tag = name
            # self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
