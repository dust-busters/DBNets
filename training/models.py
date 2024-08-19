# imports
import keras_cv
from keras.layers import LayerNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Input
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import RandomTranslation
from keras.layers import RandomFlip
from keras.layers import Concatenate
from keras.layers import GaussianNoise
from keras.layers import Add
from keras.layers import LeakyReLU
from keras_cv.layers import RandomAugmentationPipeline
from keras_cv.core import UniformFactorSampler
import tensorflow as tf
import numpy as np
import keras
import keras.ops as K


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        o_mean = y_pred[:, :1]
        o_std = y_pred[:, 1:]
        loss = tf.reduce_mean(
            tf.math.log(o_std + 1e-6)
            + tf.math.square((o_mean - y_true) / (o_std + 1e-6)) / 2
        )
        return loss


class CustomLossWarmup(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        o_mean = y_pred[:, :1]
        _ = y_pred[:, 1:]
        loss = tf.reduce_mean(tf.math.square((o_mean - y_true)))
        return loss


def inner_part_separate_mse(y_true, y_pred, i):
    d = y_pred - y_true
    square_d = K.square(d)
    return square_d[:, i]  # y has shape [batch_size, output_dim]


def separate_mse(i):
    def custom_metric_i(y_true, y_pred):
        return inner_part_separate_mse(y_true, y_pred, i)

    custom_metric_i.__name__ = f"mse_of_output_{i}"
    return custom_metric_i


########################## model with multiple parameters #############################################


def get_gaussian_beam(size, sigma, epsilon=1e-5):
    sigma = sigma + epsilon
    xy = tf.linspace(-size / 2, size / 2, size)
    xx, yy = tf.meshgrid(xy, xy)
    xx = tf.expand_dims(xx, 0)
    yy = tf.expand_dims(yy, 0)
    sigma = tf.expand_dims(sigma, 2)
    filter = (
        1
        / (2.0 * np.pi * sigma**2.0)
        * tf.math.exp(-(xx**2 + yy**2) / (2.0 * sigma**2.0))
    )
    return tf.cast(filter, tf.float32)


class RandomBeamBase(keras.layers.Layer):

    def __init__(self, maximum_res, kernel_size=65):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        self.maximum_res = maximum_res
        self.kernel_size = kernel_size

    def call(self, x, training=None):
        if training:
            sigma = tf.random.uniform(
                shape=(tf.shape(x)[0], 1), maxval=self.maximum_res
            )
            tres = self.smooth(x, sigma)
            return tres, sigma
        else:
            return x, K.convert_to_tensor([[0]])

    def smooth(self, x, sigma):
        beams = get_gaussian_beam(self.kernel_size, sigma * 128.0 / 8.0)
        tsigma = tf.cast(tf.expand_dims(tf.transpose(beams), -1), tf.float32)
        tbatch = tf.transpose(x)
        res = tf.nn.depthwise_conv2d(tbatch, tsigma, [1, 1, 1, 1], "SAME")
        tres = tf.transpose(res)
        return tres


class ResBlock(keras.Model):
    def __init__(self, kernel_n, depth=3, initializer=None):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(
            kernel_n, (3, 3), padding="same", kernel_initializer=initializer
        )
        self.activation1 = LeakyReLU()
        self.activations = []
        self.conv_layers = []
        self.depth = depth

        for i in range(1, self.depth):
            self.conv_layers += [
                Conv2D(
                    kernel_n,
                    (3, 3),
                    padding="same",
                    name=f"conv_layer_{i}",
                    kernel_initializer=initializer,
                )
            ]
            self.activations += [LeakyReLU()]

        self.add = Add()
        self.maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, inputs):
        # Apply the convolutions and activations
        x = self.conv1(inputs)
        x = self.activation1(x)
        shortcut = x

        for i in range(self.depth - 1):
            x = self.conv_layers[i](x)
            x = self.activations[i](x)

        # Add the original input (shortcut) back to the transformed output
        x = self.add([shortcut, x])

        # Apply max-pooling
        x = self.maxpool(x)
        return x


class MultiPModel(keras.Model):

    def __init__(
        self,
        act="leaky_relu",
        dropout=0.2,
        seed=0,
        noise=0,
        maximum_translation_factor=0.1,
        maximum_res=0.2,
        training=False,
        testing_resolutions=[0, 0.05, 0.1, 0.15, 0.2],
    ):
        super().__init__()
        self.augm_layers = [
            RandomTranslation(
                height_factor=(-maximum_translation_factor, maximum_translation_factor),
                width_factor=(-maximum_translation_factor, maximum_translation_factor),
                fill_mode="nearest",
            ),
            GaussianNoise(noise),
            RandomBeamBase(maximum_res),
        ]
        self.SMOOTHING_LAYER = 2
        self.norm = LayerNormalization(axis=[1, 2, 3], epsilon=1e-6)
        self.res_blocks = [ResBlock(n, initializer=None) for n in [32, 64, 128]]
        self.drop = Dropout(dropout)
        self.flatten = Flatten()
        self.dense_res = Dense(256, activation=act, input_shape=(1,))
        self.dense = [Dense(n, activation=act) for n in [256, 256, 256, 128]]
        self.out = Dense(6, activation="linear", name="o_mean")
        self.concatenate = Concatenate()
        self.training = training
        self.testing_resolutions = testing_resolutions

    def call(self, x, res=None, training=None):

        if training is None:
            training = self.training

        for i, l in enumerate(self.augm_layers):
            if i == 2:
                x, sigma = l(x, training=training)
            else:
                x = l(x, training=training)

        x = self.norm(x)

        if training:
            res = sigma

        for rb in self.res_blocks:
            x = rb(x)

        resx = self.dense_res(res)
        x = self.flatten(x)
        x = self.concatenate([x, resx])

        for dl in self.dense:
            x = self.drop(x)
            x = dl(x)

        x = self.out(x)

        return x

    def test_step(self, data):
        results = {}
        # Unpack the data
        x, y = data
        results["loss"] = 0
        for res in self.testing_resolutions:
            # generate convolved testing images
            sigma = tf.ones(shape=(tf.shape(x)[0], 1)) * res
            smoothed_x = self.augm_layers[2].smooth(x, sigma)
            # Compute predictions
            y_pred = self(smoothed_x, res=sigma, training=False)
            # Updates the metrics tracking the loss
            loss = self.compute_loss(y=y, y_pred=y_pred)
            # Update the metrics.
            for metric in self.metrics:
                if metric.name != "loss":
                    metric.update_state(y, y_pred)
                    for name, val in metric.result().items():
                        results[f"{name}_r{res}"] = val
                else:
                    metric.update_state(loss)
                    results[f"{metric.name}_r{res}"] = metric.result()
                    results["loss"] += metric.result() / len(self.testing_resolutions)
            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            for m in self.metrics:
                m.reset_state()
        return results

    def get_smoothing_layer(self):
        return self.augm_layers[self.SMOOTHING_LAYER]


def venus_multip(
    input_shape=(64, 64, 1),
    act="leaky_relu",
    dropout=0.2,
    seed=0,
    noise=0,
    maximum_res=0.2,
    n_res_blocks=1,
):

    # define the augmentation pipeline
    augm_layers = [
        RandomTranslation(
            height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="nearest"
        ),
        GaussianNoise(noise),
        RandomBeamBase(maximum_res),
    ]

    # initializer
    initializer = keras.initializers.HeNormal(seed=seed)

    # define inputs
    inputs = Input(shape=input_shape)
    input_res = Input(shape=(1,))

    # x = RandomAugmentationPipeline(augm_layers, augmentations_per_image=3, rate=0.5)(inputs)
    # augmentation and standardization

    x = LayerNormalization(axis=[1, 2])(inputs)

    for i, l in enumerate(augm_layers):
        if i == 2:
            x, sigma = l(x)
        else:
            x = l(x)

    # first block
    x = Conv2D(
        32,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c1",
        kernel_initializer=initializer,
    )(x)
    fx = Conv2D(
        32,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c2",
        kernel_initializer=initializer,
    )(x)
    fx = Conv2D(
        32,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c3",
        kernel_initializer=initializer,
    )(fx)
    x = Add()([x, fx])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b1_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # second block
    x = Conv2D(
        64,
        (3, 3),
        activation=act,
        padding="same",
        name="b2_c1",
        kernel_initializer=initializer,
    )(x)

    for i in range(n_res_blocks):
        fx = Conv2D(
            64,
            (3, 3),
            padding="same",
            activation=act,
            name=f"b2_c2_res{i}",
            kernel_initializer=initializer,
        )(x)
        fx = Conv2D(
            64,
            (3, 3),
            padding="same",
            activation=act,
            name=f"b2_c3_res{i}",
            kernel_initializer=initializer,
        )(fx)
        x = Add()([x, fx])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b2_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # third block
    x = Conv2D(
        128,
        (3, 3),
        activation=act,
        padding="same",
        name="b3_c1",
        kernel_initializer=initializer,
    )(x)
    fx = Conv2D(
        128,
        (3, 3),
        padding="same",
        activation=act,
        name="b3_c2",
        kernel_initializer=initializer,
    )(x)
    fx = Conv2D(
        128,
        (3, 3),
        padding="same",
        activation=act,
        name="b3_c3",
        kernel_initializer=initializer,
    )(fx)
    x = Add()([x, fx])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b3_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # part that takes care of the resolution
    if not training:
        sigma = input_res
    resx = Dense(256, input_shape=(1,))(sigma)

    # dense layers
    x = Flatten(name="flatten")(x)
    x = Concatenate()([x, resx])
    x = Dropout(dropout)(x)
    x = Dense(256, activation=act, name="FC1", kernel_initializer=initializer)(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation=act, name="FC2", kernel_initializer=initializer)(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation=act, name="FC3", kernel_initializer=initializer)(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation=act, name="FC4", kernel_initializer=initializer)(x)

    outLayer = Dense(6, activation="linear", name="o_mean")(x)
    simplecnn = Model(inputs, outLayer)

    return simplecnn


##########################model for ensemble###########################################################


def venus(
    input_shape=(64, 64, 1),
    act="leaky_relu",
    dropout=(0, 0, 0, 0.3, 0.3, 0.3),
    seed=0,
    noise=0,
):

    # initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    # define inputs
    inputs = Input(shape=input_shape)

    # augmentation and standardization
    x = RandomTranslation(
        height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="nearest"
    )(inputs)
    x = LayerNormalization(axis=[1, 2])(x)
    x = GaussianNoise(0.01)(x)

    # first block
    x = Conv2D(
        16,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        16,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b1_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # second block
    x = Conv2D(
        32,
        (3, 3),
        activation=act,
        padding="same",
        name="b2_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation=act,
        name="b2_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b2_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # third block
    x = Conv2D(
        64,
        (3, 3),
        activation=act,
        padding="same",
        name="b3_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation=act,
        name="b3_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b3_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # dense layers
    x = Flatten(name="flatten")(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name="FC1", kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name="FC2", kernel_initializer=initializer)(x)

    # output
    mean = Dense(1, activation="linear", name="o_mean")(x1)
    std = Dense(1, activation="softplus", name="o_std")(x1)

    outLayer = Concatenate(axis=-1)([mean, std])

    simplecnn = Model(inputs, outLayer)

    return simplecnn


def venus_RI(
    input_shape=(64, 64, 1),
    act="leaky_relu",
    dropout=(0, 0, 0, 0.3, 0.3, 0.3),
    seed=0,
    noise=0,
):

    # initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    # define inputs
    inputs = Input(shape=input_shape)

    # augmentation and standardization
    x = RandomFlip(mode="horizontal_and_vertical", seed=8365)(inputs)
    x = RandomTranslation(
        height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="nearest"
    )(x)
    x = LayerNormalization(axis=[1, 2])(x)
    x = GaussianNoise(0.01)(x)

    # first block
    x = Conv2D(
        16,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        16,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b1_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # second block
    x = Conv2D(
        32,
        (3, 3),
        activation=act,
        padding="same",
        name="b2_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation=act,
        name="b2_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b2_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # third block
    x = Conv2D(
        64,
        (3, 3),
        activation=act,
        padding="same",
        name="b3_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation=act,
        name="b3_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b3_p")(x)

    x = LayerNormalization(axis=[1, 2])(x)

    # dense layers
    x = Flatten(name="flatten")(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name="FC1", kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name="FC2", kernel_initializer=initializer)(x)

    # output
    mean = Dense(1, activation="linear", name="o_mean")(x1)
    std = Dense(1, activation="softplus", name="o_std")(x1)

    outLayer = Concatenate(axis=-1)([mean, std])

    simplecnn = Model(inputs, outLayer)

    return simplecnn


# OLD MODELS
######################################################################################################


def simpleCNNrotinvPc(
    input_shape=(64, 64, 1),
    act="leaky_relu",
    dropout=(0, 0, 0, 0.3, 0.3, 0.3),
    seed=0,
    noise=0,
):

    # initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    # define inputs
    inputs = Input(shape=input_shape)

    # normalization
    x = LayerNormalization(axis=[1, 2])(inputs)

    # noise
    # adding random noise
    x = GaussianNoise(noise, seed=(seed + 383392) % 49)(x)

    # x = SliceLayer()(x)
    x = Dropout(dropout[0])(x)
    # y = x
    # first block
    x = Conv2D(
        8,
        (2, 2),
        activation=act,
        padding="same",
        name="b1_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        8,
        (2, 2),
        activation=act,
        padding="same",
        name="b1_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b1_p")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout[1])(x)
    # x = Add()([x,y])

    # second block
    x = Conv2D(
        16,
        (5, 5),
        activation=act,
        padding="same",
        name="b2_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        16,
        (5, 5),
        padding="same",
        activation=act,
        name="b2_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b2_p")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout[2])(x)

    # third block
    x = Conv2D(
        32,
        (6, 6),
        activation=act,
        padding="same",
        name="b3_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        32,
        (6, 6),
        padding="same",
        activation=act,
        name="b3_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b3_p")(x)
    # x = BatchNormalization()(x)

    # x = CyclPoolLayer()(x)

    # dense layers
    # dense layers
    x = Flatten(name="flatten")(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name="FC1", kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name="FC2", kernel_initializer=initializer)(x)
    # x1 = Dropout(dropout[5])(x1)

    # x2 = Dense(128, activation=act, name='FC3', kernel_initializer=initializer)(x)
    # x2 = Dropout(dropout[5])(x2)

    # output
    mean = Dense(1, activation="linear", name="o_mean")(x1)
    std = Dense(1, activation="softplus", name="o_std")(x1)

    outLayer = Concatenate(axis=-1)([mean, std])
    # outLayer = Dense(2, activation='linear', name='out')(x1)

    simplecnn = Model(inputs, outLayer)

    return simplecnn


############################
def mercury(
    input_shape=(64, 64, 1),
    act="leaky_relu",
    dropout=(0, 0, 0, 0.3, 0.3, 0.3),
    seed=0,
    noise=0,
):

    # initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    # define inputs
    inputs = Input(shape=input_shape)

    # normalization
    x = LayerNormalization(axis=[1, 2])(inputs)

    # noise
    # adding random noise
    x = GaussianNoise(noise, seed=(seed + 383392) % 49)(x)

    # x = SliceLayer()(x)
    x = Dropout(dropout[0])(x)
    # y = x
    # first block
    x = Conv2D(
        16,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        16,
        (3, 3),
        activation=act,
        padding="same",
        name="b1_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b1_p")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout[1])(x)
    # x = Add()([x,y])

    # second block
    x = Conv2D(
        32,
        (5, 5),
        activation=act,
        padding="same",
        name="b2_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        32,
        (5, 5),
        padding="same",
        activation=act,
        name="b2_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b2_p")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout[2])(x)

    # third block
    x = Conv2D(
        64,
        (3, 3),
        activation=act,
        padding="same",
        name="b3_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation=act,
        name="b3_c2",
        kernel_initializer=initializer,
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b3_p")(x)
    # x = BatchNormalization()(x)

    # x = CyclPoolLayer()(x)

    # dense layers
    # dense layers
    x = Flatten(name="flatten")(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name="FC1", kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name="FC2", kernel_initializer=initializer)(x)
    # x1 = Dropout(dropout[5])(x1)

    # x2 = Dense(128, activation=act, name='FC3', kernel_initializer=initializer)(x)
    # x2 = Dropout(dropout[5])(x2)

    # output
    mean = Dense(1, activation="linear", name="o_mean")(x1)
    std = Dense(1, activation="softplus", name="o_std")(x1)

    outLayer = Concatenate(axis=-1)([mean, std])
    # outLayer = Dense(2, activation='linear', name='out')(x1)

    simplecnn = Model(inputs, outLayer)

    return simplecnn


###################


def ResNet3(
    input_shape=(64, 64, 1),
    act="leaky_relu",
    dropout=(0, 0, 0, 0.3, 0.3, 0.3),
    seed=0,
    noise=0,
):

    # initializer
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    # define inputs
    inputs = Input(shape=input_shape)

    # normalization
    x = LayerNormalization(axis=[1, 2])(inputs)

    # noise
    # adding random noise
    x = GaussianNoise(noise, seed=(seed + 383392) % 49)(x)

    # x = SliceLayer()(x)
    x = Dropout(dropout[0])(x)
    x = Conv2D(
        8,
        (2, 2),
        activation=act,
        padding="same",
        name="b1_c0",
        kernel_initializer=initializer,
    )(x)
    y = x
    # first block
    x = Conv2D(
        8,
        (2, 2),
        activation=act,
        padding="same",
        name="b1_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        8,
        (2, 2),
        activation=act,
        padding="same",
        name="b1_c2",
        kernel_initializer=initializer,
    )(x)
    # x = BatchNormalization()(x)
    # x = Dropout(dropout[1])(x)
    x = Add()([x, y])

    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b1_p")(x)

    x = Conv2D(
        16,
        (5, 5),
        activation=act,
        padding="same",
        name="b2_c0",
        kernel_initializer=initializer,
    )(x)
    y = x
    # second block
    x = Conv2D(
        16,
        (5, 5),
        activation=act,
        padding="same",
        name="b2_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        16,
        (5, 5),
        padding="same",
        activation=act,
        name="b2_c2",
        kernel_initializer=initializer,
    )(x)
    x = Add()([x, y])
    # x = BatchNormalization()(x)
    # x = Dropout(dropout[2])(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b2_p")(x)

    x = Conv2D(
        32,
        (6, 6),
        activation=act,
        padding="same",
        name="b3_c0",
        kernel_initializer=initializer,
    )(x)
    y = x
    # third block
    x = Conv2D(
        32,
        (6, 6),
        activation=act,
        padding="same",
        name="b3_c1",
        kernel_initializer=initializer,
    )(x)
    x = Conv2D(
        32,
        (6, 6),
        padding="same",
        activation=act,
        name="b3_c2",
        kernel_initializer=initializer,
    )(x)
    x = Add()([x, y])
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="b3_p")(x)

    # dense layers
    # dense layers
    x = Flatten(name="flatten")(x)
    x = Dropout(dropout[3])(x)
    x = Dense(256, activation=act, name="FC1", kernel_initializer=initializer)(x)
    x = Dropout(dropout[4])(x)
    x1 = Dense(128, activation=act, name="FC2", kernel_initializer=initializer)(x)
    # x1 = Dropout(dropout[5])(x1)

    # x2 = Dense(128, activation=act, name='FC3', kernel_initializer=initializer)(x)
    # x2 = Dropout(dropout[5])(x2)

    # output
    mean = Dense(1, activation="linear", name="o_mean")(x1)
    std = Dense(1, activation="softplus", name="o_std")(x1)

    outLayer = Concatenate(axis=-1)([mean, std])
    # outLayer = Dense(2, activation='linear', name='out')(x1)

    simplecnn = Model(inputs, outLayer)

    return simplecnn
