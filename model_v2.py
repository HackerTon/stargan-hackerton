import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.utils.version_utils import training
from tensorflow.python.ops.gen_data_flow_ops import map_incomplete_size
from tensorflow.python.ops.linalg_ops import norm
from tensorflow.python.ops.nn_impl import normalize


class InstanceNorm(keras.layers.Layer):
    def __init__(self, affine=True):
        super().__init__()

        if affine:
            self.gamma = tf.Variable(1.0)
            self.beta = tf.Variable(0.0)
            self.affinetrans = lambda x: x * self.gamma + self.beta
        else:
            self.affinetrans = lambda x: x

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        output = (x - mean) / tf.maximum(variance, 1e-6)

        return self.affinetrans(output)


class AdaptiveNorm(tf.Module):
    def __init__(self, name='AdaptiveNorm', nfeatures=64):
        super().__init__(name=name)
        self.norm = InstanceNorm()
        self.style = tf.keras.layers.Dense(nfeatures * 2)

    def __call__(self, x, s):
        style = self.style(s)

        normalized = self.norm(x)
        w = tf.split(style, num_or_size_splits=2,
                     axis=1)  # (bs, nfeatures * 2)

        return (1 + w[:, 0]) * normalized + w[:, 1]  # (bs, h, w, nfeatures)


class Resblk(keras.layers.Layer):
    def __init__(self, filters=128, downsample=False, normalize=False, affine=True):
        super().__init__()
        self.layers = []
        self.shortcut = []
        self.nfilter = filters
        self.downsample = downsample
        self.normalize = normalize
        self.affine = affine

    def build(self, input_shape: tf.TensorShape):
        chandim = input_shape[-1]

        if self.normalize:
            self.layers.append(InstanceNorm(affine=self.affine))

        self.layers.append(keras.layers.LeakyReLU(0.2))
        self.layers.append(keras.layers.Conv2D(chandim, 3, padding='same'))

        self.layers.append(keras.layers.LeakyReLU(0.2))
        self.layers.append(keras.layers.Conv2D(
            self.nfilter, 3, padding='same'))

        if input_shape[-1] != self.nfilter:
            self.shortcut.append(keras.layers.Conv2D(
                self.nfilter, 1, padding='valid'))
        else:
            self.shortcut.append(keras.layers.Activation('linear'))

        if self.downsample:
            self.layers.append(keras.layers.AvgPool2D())
            self.shortcut.append(keras.layers.AvgPool2D())

    def call(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        outshort = inputs
        for layer in self.shortcut:
            outshort = layer(outshort)

        outputs = outputs + outshort
        return outputs


def create_mapping_head() -> keras.Sequential:
    head = keras.Sequential()
    head.add(keras.layers.InputLayer([512]))

    for _ in range(3):
        head.add(keras.layers.Dense(512, keras.activations.relu))
    head.add(keras.layers.Dense(64))

    return head


def create_mapping_network() -> keras.Sequential:
    mapping = keras.Sequential()
    mapping.add(keras.layers.InputLayer([16]))

    for _ in range(4):
        mapping.add(keras.layers.Dense(512, keras.activations.relu))

    return mapping


class Mapping(keras.Model):
    def __init__(self, num_head=3, **kargs):
        super().__init__(**kargs)
        self.mlp = create_mapping_network()
        self.head = []

        for _ in range(num_head):
            self.head.append(create_mapping_head())

    def __call__(self, latent, idxhead):
        out_latent = self.mlp(latent)
        out_head = self.head[idxhead](out_latent)
        return out_head


def create_style_network() -> keras.Sequential:
    stylenet = keras.Sequential()
    stylenet.add(keras.layers.InputLayer([256, 256, 3]))
    stylenet.add(keras.layers.Conv2D(64, 1, 1, 'same'))
    stylenet.add()


def create_shared() -> keras.Sequential:
    net = keras.Sequential(name='shared')
    net.add(keras.layers.InputLayer([256, 256, 3]))
    net.add(keras.layers.Conv2D(64, 1, 1, padding='same'))
    net.add(Resblk(128, True))
    net.add(Resblk(256, True))

    for _ in range(4):
        net.add(Resblk(512, True))

    net.add(keras.layers.LeakyReLU(0.2))
    net.add(keras.layers.Conv2D(512, 4))
    net.add(keras.layers.LeakyReLU(0.2))

    return net


class Discriminator(keras.Model):
    def __init__(self, K, **kargs):
        super().__init__(**kargs)

        self.model = create_shared()
        self.head = []
        self.reshape = keras.layers.Reshape([512])

        for _ in range(K):
            self.head.append(keras.layers.Dense(1))

    def call(self, inputs, head):
        outputs = self.model(inputs)
        outputs = self.reshape(outputs)
        return self.head[head](outputs)


class Styleencoder(keras.Model):
    def __init__(self, K, D=64, **kargs):
        super().__init__(**kargs)
        self.model = create_shared()
        self.head = []
        self.reshape = keras.layers.Reshape([512])

        for _ in range(K):
            self.head.append(keras.layers.Dense(D))

    def call(self, inputs, head):
        outputs = self.model(inputs)
        outputs = self.reshape(outputs)
        return self.head[head](outputs)


class Generator(keras.Model):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.model = create_generator()

    def call(self, inputs, latent):
        pass
