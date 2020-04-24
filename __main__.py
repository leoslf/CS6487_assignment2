import sys
import os
import numpy as np
import logging

from regression.baseline_model import *
from regression.utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

basenames = ["train", "validation", "test"]
dataset_dir = "social-checkin-prediction"
filenames = list(map(lambda basename: os.path.join(dataset_dir, "%s.csv" % basename), basenames))

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    
    dataset_loader = compose(split_XY, drop_userid, load_csv)

    (train_X, train_Y), (validation_X, validation_Y), (test_X, test_Y) = list(map(dataset_loader, filenames))

    logger.debug("train_X: %r, train_Y: %r", train_X.shape, train_Y.shape)
    logger.debug("validation_X: %r, validation_Y: %r", validation_X.shape, validation_Y.shape)
    logger.debug("test_X: %r, test_Y: %r", test_X.shape, test_Y.shape)


    baseline = BaselineModel() 
    baseline.fit(train_X, train_Y, validation_X, validation_Y)
    print (baseline.evaluate(test_X, test_Y))

    modified_baseline = ModifiedBaselineModel()
    modified_baseline.fit(train_X, train_Y, validation_X, validation_Y)
    print (modified_baseline.evaluate(test_X, test_Y))

