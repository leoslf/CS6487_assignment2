from regression.model_base import *

class BaselineModel(BaseModel):
    @property
    def time_activation(self):
        return "linear"

    @property
    def apply_batchnorm(self):
        return False

    @property
    def kernel_regularizer(self):
        return None

    @property
    def dropout_rate(self):
        return 0
    
    def prepare_model(self):
        inputs = Input(shape = self.input_shape)
        conv_1 = Conv1D(64, 2, input_shape=self.input_shape, activation="relu", kernel_regularizer=self.kernel_regularizer)(inputs)
        if self.apply_batchnorm:
            conv_1 = BatchNorm()(conv_1)
        conv_1 = Activation("relu")(conv_1)

        conv_2 = Conv1D(32, 2, kernel_regularizer=self.kernel_regularizer)(conv_1)
        if self.apply_batchnorm:
            conv_2 = BatchNorm()(conv_2)
        conv_2 = Activation("relu")(conv_2)

        conv_3 = Conv1D(32, 1, kernel_regularizer=self.kernel_regularizer)(conv_2)
        if self.apply_batchnorm:
            conv_3 = BatchNorm()(conv_3)
        conv_3 = Activation("relu")(conv_3)

        flatten = Flatten()(conv_3)

        dense_1 = Dense(32, activation="relu", kernel_regularizer=self.kernel_regularizer)(flatten)
        dropout_1 = Dropout(self.dropout_rate)(dense_1)

        dense_2 = Dense(64, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_1)
        dropout_2 = Dropout(self.dropout_rate)(dense_2)

        dense_3 = Dense(32, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_2)
        dropout_3 = Dropout(self.dropout_rate)(dense_3)

        dense_4 = Dense(16, activation="relu", kernel_regularizer=self.kernel_regularizer)(dropout_3)

        coordinate = Dense(2, activation="linear", kernel_regularizer=self.kernel_regularizer)(dense_4)
        time_component = Dense(1, activation=self.time_activation, kernel_regularizer=self.kernel_regularizer)(dense_4)

        outputs = merge.concatenate([coordinate, time_component])

        return Model(inputs, outputs)


class ModifiedBaselineModel(BaselineModel):
    @property
    def time_activation(self):
        def real_mod(x):
            return x % 1
        return real_mod
    
    @property
    def loss(self):
        mse = MeanSquaredError()
        coord_weight = 0.5
        time_weight = 0.5
        def custom_loss(y_true, y_predict):
            coord_true, coord_predict = y_true[:, :2], y_predict[:, :2]
            time_true, time_predict = y_true[:, 2], y_predict[:, 2]

            def custom(y_true, y_predict):
                abs_diff = K.abs(y_true - y_predict)
                diff_int = tf.floor(abs_diff)
                diff = abs_diff - diff_int - 0.5
                lessthan_0_5 = K.cast(K.less_equal(diff, -0.5), tf.float32)

                error = lessthan_0_5 * (diff + 1) + (1 - lessthan_0_5) * diff

                return K.mean(K.abs(error))

            # MSE for coordinates, custom for time
            return coord_weight * K.mean(K.square(coord_true - coord_predict)) + time_weight * custom(time_true, time_predict)
        return custom_loss

