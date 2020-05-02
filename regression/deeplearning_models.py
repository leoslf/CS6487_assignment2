from regression.regularized_models import *

class BatchNorm(RegularizedModel_L2):
    @property
    def apply_batchnorm(self):
        return True


class ResNet(BatchNorm):
    def relu_bn(self, X):
        X = ReLU()(X)
        # NOTE: always True as inherited from BatchNorm
        if self.apply_batchnorm:
            X = BatchNormalization()(X)
        return X

    def residual_block(self, X, num_filters, kernel_size = 3, downsampling = False):
        tilde_X = Conv1D(num_filters,
                         kernel_size = kernel_size,
                         strides = (2 if downsampling else 1),
                         padding = "same",
                         kernel_regularizer = self.kernel_regularizer)(X)
        tilde_X = self.relu_bn(tilde_X)
        
        tilde_X = Conv1D(num_filters,
                         kernel_size = kernel_size,
                         strides = 1,
                         padding = "same",
                         kernel_regularizer = self.kernel_regularizer)(tilde_X)
        if downsampling:
            X = Conv1D(num_filters,
                       kernel_size = 1,
                       strides = 2,
                       padding = "same",
                       kernel_regularizer = self.kernel_regularizer)(X)

        result = Add()([X, tilde_X])
        result = self.relu_bn(result)

        return result

    @property
    def blocks_spec(self):
        return [2, 3, 2]

    @property
    def filter_growth(self):
        return 2

    def residual_module(self, inputs, num_filters, filter_growth):
        residual = inputs
        for i, num_blocks in enumerate(self.blocks_spec):
            for j in range(num_blocks):
                downsampling = (i > 0 and j == 0)
                residual = self.residual_block(residual, num_filters, downsampling = downsampling)
            num_filters += filter_growth
        return residual, num_filters


    def prepare_model(self):
        inputs = Input(shape = self.input_shape)
        # First Perform BatchNormalization
        bn = BatchNormalization()(inputs)

        num_filters = 64
        
        conv_1 = Conv1D(num_filters, 2, kernel_regularizer=self.kernel_regularizer)(bn)
        conv_1 = self.relu_bn(conv_1)

        # renaming
        residual, num_filters = self.residual_module(conv_1, num_filters, self.filter_growth)

        pool = AveragePooling1D(1)(residual)

        flatten = Flatten()(pool)

        # dense_1 = Dense(32, activation="relu", kernel_regularizer=self.kernel_regularizer)(flatten)
        # dropout_1 = Dropout(self.dropout_rate)(dense_1)

        # dense_2 = Dense(64, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_1)
        # dropout_2 = Dropout(self.dropout_rate)(dense_2)

        # dense_3 = Dense(32, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_2)
        # dropout_3 = Dropout(self.dropout_rate)(dense_3)

        # dense_4 = Dense(16, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_3)

        # coordinate = Dense(2, activation="linear", kernel_regularizer=self.kernel_regularizer)(dense_4)
        # time_component = Dense(1, activation=self.time_activation, kernel_regularizer=self.kernel_regularizer)(dense_4)

        coordinate = Dense(2, activation="linear", kernel_regularizer=self.kernel_regularizer)(flatten)
        time_component = Dense(1, activation=self.time_activation, kernel_regularizer=self.kernel_regularizer)(flatten)

        outputs = merge.concatenate([coordinate, time_component])

        return Model(inputs, outputs)

class DenseNet(BatchNorm):
    def H(self, inputs, num_filters, kernel_size = 3):
        """ Composition of (Convolution, ReLU, BatchNormalization) """
        X = BatchNormalization()(inputs)
        X = Activation("relu")(X)
        X = ZeroPadding1D(1)(X)
        X = Conv1D(num_filters,
                   kernel_size = kernel_size,
                   use_bias = False,
                   kernel_initializer = "he_normal",
                   kernel_regularizer = self.kernel_regularizer)(X)
        X = Dropout(self.dropout_rate)(X)
        return X
    
    @property
    def compression(self):
        return 0.8

    def transition(self, inputs, num_filters, compression):
        X = BatchNormalization()(inputs)
        X = Activation("relu")(X)

        feature_map_dimensions = int(inputs.shape[1])
        X = Conv1D(np.floor(compression * feature_map_dimensions).astype(np.int),
                   kernel_size = 1,
                   use_bias = False,
                   padding = "same",
                   kernel_initializer = 'he_normal',
                   kernel_regularizer = self.kernel_regularizer)(X)
        X = Dropout(self.dropout_rate)(X)
        X = AveragePooling1D(1)(X)
        return X

    def dense_block(self, inputs, num_layers, num_filters, filter_growth):
        for i in range(num_layers):
            outputs = self.H(inputs, num_filters)
            inputs = Concatenate()([outputs, inputs])
            num_filters += filter_growth

        return inputs, num_filters

    @property
    def blocks_spec(self):
        return [2, 3, 3, 2]

    @property
    def filter_growth(self):
        return 2

    def dense_module(self, X, num_filters, filter_growth):
        for i, num_layers in enumerate(self.blocks_spec):
            X, num_filters = self.dense_block(X, num_layers, num_filters, filter_growth)
            X = self.transition(X, num_filters, self.compression)
        return X, num_filters

    def prepare_model(self):
        inputs = Input(shape = self.input_shape)

        num_filters = 64

        conv_1 = Conv1D(num_filters, 2, input_shape=self.input_shape, kernel_regularizer=self.kernel_regularizer)(inputs)

        densenet, num_filters = self.dense_module(conv_1, num_filters, self.filter_growth)

        # flatten = Flatten()(densenet)
        pool = GlobalAveragePooling1D()(densenet)

        # dense_1 = Dense(32, activation="relu", kernel_regularizer=self.kernel_regularizer)(flatten)
        # dropout_1 = Dropout(self.dropout_rate)(dense_1)

        # dense_2 = Dense(64, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_1)
        # dropout_2 = Dropout(self.dropout_rate)(dense_2)

        # dense_3 = Dense(32, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_2)
        # dropout_3 = Dropout(self.dropout_rate)(dense_3)

        # dense_4 = Dense(16, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_3)

        # coordinate = Dense(2, activation="linear", kernel_regularizer=self.kernel_regularizer)(dense_4)
        # time_component = Dense(1, activation=self.time_activation, kernel_regularizer=self.kernel_regularizer)(dense_4)

        dense_4 = Dense(16, activation="relu", kernel_regularizer=self.kernel_regularizer)(pool)

        coordinate = Dense(2, activation="linear", kernel_regularizer=self.kernel_regularizer)(dense_4)
        time_component = Dense(1, activation=self.time_activation, kernel_regularizer=self.kernel_regularizer)(dense_4)

        outputs = merge.concatenate([coordinate, time_component])

        return Model(inputs, outputs)
