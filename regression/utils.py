import numpy as np
import logging

import functools

from itertools import *
from datetime import datetime


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
    X = dataset[:, :num_features * X_records]
    Y = dataset[:, num_features * X_records:]
    return X, Y
