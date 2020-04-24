from regression.model_base import *

class BaselineModel(BaseModel):
    def prepare_model(self):
        inputs = Input(shape = self.input_shape)
        dense_1 = Dense(64, input_shape=self.input_shape, activation="relu")(inputs)
        dense_2 = Dense(512, activation="relu")(dense_1)
        dense_3 = Dense(256, activation="relu")(dense_2)
        dense_4 = Dense(32, activation="relu")(dense_3)
        coordinate = Dense(2, activation="linear")(dense_4)
        time_component = Dense(1, activation="linear")(dense_4)

        outputs = merge.concatenate([coordinate, time_component])

        return Model(inputs, outputs)


class ModifiedBaselineModel(BaselineModel):
    def prepare_model(self):
        inputs = Input(shape = self.input_shape)
        dense_1 = Dense(64, input_shape=self.input_shape, activation="relu")(inputs)
        dense_2 = Dense(512, activation="relu")(dense_1)
        dense_3 = Dense(256, activation="relu")(dense_2)
        dense_4 = Dense(32, activation="relu")(dense_3)
        coordinate = Dense(2, activation="linear")(dense_4)
        time_component = Dense(1, activation="sigmoid")(dense_4)

        outputs = merge.concatenate([coordinate, time_component])

        return Model(inputs, outputs)
    
    @property
    def loss(self):
        mse = MeanSquaredError()
        def custom_loss(y_true, y_predict):
            weights = np.array([0.5, 0.5])
            coord_true, coord_predict = y_true[:, :2], y_predict[:, :2]
            time_true, time_predict = y_true[:, 2], y_predict[:, 2]

            def custom(y_true, y_predict):
                abs_diff = K.abs(y_true - y_predict)
                diff_int = tf.floor(abs_diff)
                diff = abs_diff - diff_int - 0.5
                lessthan_0_5 = K.cast(K.less_equal(diff, -0.5), tf.float32)

                return lessthan_0_5 * (diff + 1) + (1 - lessthan_0_5) * diff

            # MSE for coordinates, custom for time
            return weights * [K.mean(K.square(coord_true - coord_predict)), custom(time_true, time_predict)]
        return custom_loss



