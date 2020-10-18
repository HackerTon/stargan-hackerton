from threading import Semaphore
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.utils.version_utils import training
from tensorflow.python.ops.gen_data_flow_ops import map_incomplete_size
from tensorflow.python.ops.linalg_ops import norm
from tensorflow.python.ops.nn_impl import normalize
from tensorflow.python.training.tracking.base import no_manual_dependency_tracking_scope


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


class AdaptiveNorm(keras.layers.Layer):
    def __init__(self, nfeatures=64, **kargs):
        super().__init__(**kargs)
        self.nfeatures = nfeatures
        self.norm = InstanceNorm()

    def build(self, input_shape: tf.TensorShape):
        self.chandims = input_shape[0][-1]
        self.beta = keras.layers.Dense(self.chandims, input_shape=[64])
        self.gamma = keras.layers.Dense(self.chandims, input_shape=[64])

    def call(self, inputs):
        beta = tf.reshape(self.beta(inputs[1]), [-1, 1, 1, self.chandims])
        gamma = tf.reshape(self.gamma(inputs[1]), [-1, 1, 1, self.chandims])
        normalized = self.norm(inputs[0])

        return (1 + gamma) * normalized + beta  # (bs, h, w, nfeatures)


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
        self.layers.append(keras.layers.Conv2D(
            self.nfilter, 3, padding='same'))

        if self.downsample:
            self.layers.append(keras.layers.AvgPool2D())

        if chandim != self.nfilter:
            self.shortcut.append(keras.layers.Conv2D(
                self.nfilter, 1, padding='valid'))
        else:
            self.shortcut.append(keras.layers.Activation('linear'))

        if self.downsample:
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


class Adaresblk(Resblk):
    def __init__(self, filters=128, upsample=False, normalize=False, affine=True, **kargs):
        super().__init__(filters, normalize=normalize, affine=affine, **kargs)
        self.upsample = upsample

    def build(self, input_shape: tf.TensorShape):
        chandim = input_shape[0][-1]

        if self.normalize:
            self.normalize1 = AdaptiveNorm()
            self.normalize2 = AdaptiveNorm()
        else:
            self.normalize1 = keras.layers.Activation('linear')
            self.normalize2 = keras.layers.Activation('linear')

        self.lrelu1 = keras.layers.LeakyReLU(0.2)

        if self.upsample:
            self.upsample1 = keras.layers.UpSampling2D()
            self.upsample2 = keras.layers.UpSampling2D()
        else:
            self.upsample1 = keras.layers.Activation('linear')
            self.upsample2 = keras.layers.Activation('linear')

        self.conv1 = keras.layers.Conv2D(
            chandim, 3, padding='same')

        self.lrelu2 = keras.layers.LeakyReLU(0.2)

        self.conv2 = keras.layers.Conv2D(
            self.nfilter, 3, padding='same')

        if chandim != self.nfilter:
            self.shortcut = keras.layers.Conv2D(
                self.nfilter, 1, padding='valid')
        else:
            self.shortcut = keras.layers.Activation('linear')

    def call(self, inputs):  # [images, style]
        outputs = self.normalize1(inputs)
        outputs = self.lrelu1(outputs)
        outputs = self.upsample1(outputs)
        outputs = self.conv1(outputs)
        outputs = self.normalize2([outputs, inputs[1]])
        outputs = self.lrelu2(outputs)
        outputs = self.conv2(outputs)
        shortcuts = self.shortcut(self.upsample2(inputs[0]))

        return outputs + shortcuts


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
        self.model = []
        self.model2 = []

        self.model.append(keras.layers.Conv2D(64, 1, padding='same'))

        self.model.append(Resblk(128, True, True))
        self.model.append(Resblk(256, True, True))
        self.model.append(Resblk(512, True, True))
        self.model.append(Resblk(512, True, True))

        self.model.append(Resblk(512, normalize=True))
        self.model.append(Resblk(512, normalize=True))

        self.model2.append(Adaresblk(512, normalize=True))
        self.model2.append(Adaresblk(512, normalize=True))
        self.model2.append(Adaresblk(512, True, True))
        self.model2.append(Adaresblk(256, True, True))
        self.model2.append(Adaresblk(128, True, True))
        self.model2.append(Adaresblk(64, True, True))
        self.convf = keras.layers.Conv2D(3, 1, padding='same')

    def call(self, inputs):  # inputs [images, style]
        outputs = inputs[0]
        for layer in self.model:
            outputs = layer(outputs)
        for layer in self.model2:
            outputs = layer([outputs, inputs[1]])
        outputs = self.convf(outputs)

        return outputs
