import sys
import os
import operator

import numpy as np
import logging
import pickle

from datetime import datetime

from regression.baseline_models import *
from regression.regularized_models import *
from regression.deeplearning_models import *
from regression.utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

basenames = ["train", "validation", "test"]
dataset_dir = "social-checkin-prediction"
filenames = list(map(lambda basename: os.path.join(dataset_dir, "%s.csv" % basename), basenames))

models = [
    "BaselineModel",
    "ModifiedBaselineModel",
    "RegularizedModel_EarlyStopping",
    "RegularizedModel_L2",
    "RegularizedModel_L1",
    "RegularizedModel_Dropout",
    "RegularizedModel_Ensembling",
    "RegularizedModel_DataAugmentation",
    "BatchNorm",
    "ResNet",
    "DenseNet",
]

def get_model(name, *argv, **kwargs):
    return globals()[name](*argv, **kwargs)

if __name__ == "__main__":
    sys.setrecursionlimit(15000)
    np.set_printoptions(suppress=True)
    
    dataset_loader = compose(split_XY, drop_userid, load_csv)

    (train_X, train_Y), (validation_X, validation_Y), (test_X, test_Y) = list(map(dataset_loader, filenames))

    logger.debug("train_X: %r, train_Y: %r", train_X.shape, train_Y.shape)
    logger.debug("validation_X: %r, validation_Y: %r", validation_X.shape, validation_Y.shape)
    logger.debug("test_X: %r, test_Y: %r", test_X.shape, test_Y.shape)

    losses = {}

    for model in map(get_model, models):
        # history = model.fit(train_X, train_Y, validation_X, validation_Y)
        # with open("%s_history_%s.pickle" % (model.name, datetime.now().strftime("%Y%m%d-%H%M%S")), "wb") as f:
        #     pickle.dump(history, f)

        test_loss = model.evaluate(test_X, test_Y)
        logger.info("model \"%s\": testing loss: %f", model.name, test_loss)
        losses[model.name] = test_loss


    with open("losses.pickle", "wb") as f:
        pickle.dump(losses, f)

