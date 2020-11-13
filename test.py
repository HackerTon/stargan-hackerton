import numpy as np
from ops import create_dataset, create_dataset_celb, r1_loss
import unittest

import tensorflow as tf
from tensorflow.python.keras.utils.version_utils import ModelVersionSelector
from tensorflow.python.ops.functional_ops import Gradient

import model as mdl
import model_v2
import train


class Outputtest(unittest.TestCase):
    def testgenout(self):
        model = mdl.generator(2)

        sampleimg = tf.random.uniform([1, 128, 128, 3])
        samplelabel = tf.random.uniform(
            [1, 128, 128, 2], maxval=2, dtype=tf.int32)

        output = model([sampleimg, samplelabel])

        self.assertEqual(tf.TensorShape([1, 128, 128, 3]), tf.shape(output))

    def testdisout(self):
        model = model_v2.Discriminator(4)
        sample = tf.random.uniform([5, 256, 256, 3])
        domain = tf.random.uniform([5, 1], maxval=4, dtype=tf.int32)

        outputs = model(sample, domain)
        self.assertEqual(tf.TensorShape([5, 1]), tf.shape(outputs))

    def testmapout(self):
        mapping = model_v2.Mapping(4)
        correctarr = tf.TensorShape([5, 64])
        latent = tf.random.normal([5, 16])
        domain = tf.random.uniform([5, 1], maxval=4, dtype=tf.int32)

        self.assertEqual(correctarr, tf.shape(mapping(latent, domain)))

    def teststyleout(self):
        styleencoder = model_v2.Styleencoder(4)
        correct = tf.TensorShape([5, 64])
        sample = tf.random.normal([5, 256, 256, 3])
        domain = tf.random.uniform([5, 1], maxval=4, dtype=tf.int32)

        self.assertEqual(correct, tf.shape(styleencoder(sample, domain)))

    def testresblk(self):
        resblock = model_v2.Resblk(
            filters=256, downsample=True, normalize=True)

        shape = tf.TensorShape([1, 128, 128, 256])
        sample = tf.random.uniform([1, 256, 256, 3])
        output = resblock(sample)

        self.assertEqual(shape, tf.shape(output))

    def testadaresblk(self):
        resblock = model_v2.Adaresblk(
            filters=256, upsample=True, normalize=True)

        shape = tf.TensorShape([1, 256, 256, 256])
        sample = tf.random.uniform([1, 128, 128, 3])
        style = tf.random.uniform([1, 64])
        output = resblock([sample, style])

        self.assertEqual(shape, tf.shape(output))

    def testadain(self):
        adain = model_v2.AdaptiveNorm()
        shape = tf.TensorShape([1, 128, 128, 3])

        sample = tf.random.uniform([1, 128, 128, 3])
        style = tf.random.uniform([1, 64])
        outputs = adain([sample, style])

        self.assertEqual(shape, tf.shape(outputs))

    def testgenerator(self):
        generator = model_v2.Generator()

        shape = tf.TensorShape([1, 256, 256, 3])
        sample = tf.random.uniform([1, 256, 256, 3])
        style = tf.random.uniform([1, 64])
        output = generator(sample, style)

        self.assertEqual(shape, tf.shape(output))

    def testr1loss(self):
        dis = model_v2.Discriminator(5)
        sample = tf.random.uniform([5, 256, 256, 3])
        domain = tf.random.uniform([5, 1], maxval=4, dtype=tf.int32)
        shape = tf.TensorShape([])

        out = r1_loss(dis, sample, domain)
        self.assertEqual(shape, tf.shape(out))

    def gradtest(self):
        zeros1 = np.zeros([1, 256, 256, 3])
        zeros2 = np.zeros([1])
        ds = tf.data.Dataset.from_tensor_slices((zeros1, zeros2))
