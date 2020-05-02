import logging

from regression.baseline_models import *

logger = logging.getLogger(__name__)

class RegularizedModel_EarlyStopping(ModifiedBaselineModel):
    @property
    def use_earlystopping(self):
        return True

class RegularizedModel_L2(RegularizedModel_EarlyStopping):
    @property
    def kernel_regularizer(self):
        return l2(0.0002)

class RegularizedModel_L1(RegularizedModel_EarlyStopping):
    @property
    def kernel_regularizer(self):
        return l1(0.0002)

class RegularizedModel_Dropout(RegularizedModel_EarlyStopping):
    @property
    def dropout_rate(self):
        return 0.4


class RegularizedModel_Ensembling(RegularizedModel_EarlyStopping):
    """ Ensembling with EarlyStopping, L1, L2 """
    def load_weights(self):
        models = [RegularizedModel_EarlyStopping(), RegularizedModel_L1(), RegularizedModel_L2()]
        weights = [model.model.get_weights() for model in models]
        new_weights = [[np.array(w).mean(axis = 0) for w in zip(*W)] for W in zip(*weights)]
        self.model.set_weights(new_weights)

    def fit(self, *argv, **kwargs):
        self.save_weights()
        return None


class RegularizedModel_DataAugmentation(RegularizedModel_EarlyStopping):
    """ Adding Gaussian Noise into t """
    def fit(self, trainX, trainY, *argv, sigma = 0.01, **kwargs):
        Z = np.random.randn(len(trainX), 1) * sigma
        logger.info("Z.shape: %r", Z.shape)
        
        positive_X = trainX.copy()
        positive_X[:, :, 2] += Z
        positive_Y = trainY.copy()
        positive_Y[:, 2] += Z.flatten()
        
        # negative_X = trainX.copy()
        # negative_X[:, :, 2] -= Z
        # negative_Y = trainY.copy()
        # negative_Y[:, 2] -= Z.flatten()

        augmented_trainX = np.row_stack((positive_X, trainX)) # , negative_X))
        augmented_trainY = np.row_stack((positive_Y, trainY)) # , negative_Y))
        return super().fit(augmented_trainX, augmented_trainY, *argv, **kwargs)





