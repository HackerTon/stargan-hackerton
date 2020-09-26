import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils.version_utils import training
from tensorflow.python.ops.linalg_ops import norm
from tensorflow.python.ops.nn_impl import normalize


class InstanceNorm(tf.Module):
    def __init__(self, name=None, affine=False):
        super().__init__(name=name)
        self.affine = affine
        self.gamma = tf.Variable(1.0)
        self.beta = tf.Variable(0.0)

    def __call__(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        output = (x - mean) / tf.maximum(variance, 1e-6)

        if self.affine:
            output = self.gamma * output
            output += self.beta

        return output


class AdaptiveNorm(tf.Module):
    def __init__(self, name=None, nfeatures=64):
        super().__init__(name=name)
        self.norm = InstanceNorm()
        self.style = tf.keras.layers.Dense(nfeatures * 2)

    def __call__(self, x, s):
        style = self.style(s)

        normalized = self.norm(x)
        w = tf.split(style, num_or_size_splits=2,
                     axis=1)  # (bs, nfeatures * 2)

        return (1 + w[:, 0]) * normalized + w[:, 1]  # (bs, h, w, nfeatures)


def resblock_v1(inputs, filters=128, downsample=False, normalize=False):
    output = inputs

    if normalize:
        output = InstanceNorm(affine=True)(output)

    input_dim = inputs.shape[-1]

    output = tf.keras.layers.LeakyReLU(0.2)(output)
    output = tf.keras.layers.Conv2D(filters=input_dim, kernel_size=3,
                                    strides=1, padding='same')(output)

    if downsample:
        output = tf.keras.layers.AvgPool2D()(output)

    if normalize:
        output = InstanceNorm(affine=True)(output)

    output = tf.keras.layers.LeakyReLU(0.2)(output)
    output = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                    strides=1, padding='same')(output)

    residual = inputs
    if input_dim != filters:
        residual = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                          strides=1, padding='valid')(inputs)

    if downsample:
        residual = tf.keras.layers.AvgPool2D()(residual)

    return tf.keras.layers.Add()([output, residual])


class Generator(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.generator = Generator._build_generator()
        self.generator.summary()

    @staticmethod
    def _build_generator():
        inputs = tf.keras.Input([256, 256, 3])

        output = tf.keras.layers.Conv2D(64, 1, 1, padding='same')(inputs)

        for i in [128, 256, 512, 512]:
            output = resblock_v1(output, i, True, False)

        for i in [512, 512]:
            output = resblock_v1(output, i, False, True)

        return tf.keras.Model(inputs=[inputs], outputs=[output])

    def __call__(self, inputs, domain):
        """
        inputs = image
        domain = 1 - 2
        """
        pass


# Generator()
