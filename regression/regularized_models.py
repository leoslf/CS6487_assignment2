from regression.baseline_models import *

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


# TODO
class RegularizedModel_Ensembling(RegularizedModel_EarlyStopping):
    pass

class RegularizedModel_DataAugmentation(RegularizedModel_EarlyStopping):
    pass



