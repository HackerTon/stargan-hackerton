import argparse
import datetime
import logging
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import data

import model
import model_v2
from helper.benchmark import benchmark_dataset
from ops import create_dataset, create_dataset_celb, label2onehot_C

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.random.set_seed(1234567)


def train(args):
    batch_size = int(args.bs)

    logger = logging.getLogger(__name__)
    dataset = create_dataset_celb(args.dir)
    batch_ds = dataset.shuffle(1000).batch(
        batch_size, drop_remainder=True)

    suffix = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = os.path.join('logdir', suffix, 'train')
    summarywriter = tf.summary.create_file_writer(filename)

    stargan = model.Stargan(summarywriter, 128, 2)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               generator=stargan.generator,
                               discrimintor=stargan.discriminator,
                               genopti=stargan.genopti,
                               disopti=stargan.disopti,
                               dataset=batch_ds
                               )

    ckptm = tf.train.CheckpointManager(ckpt, 'checkpoint', 3)
    ckptm.restore_or_initialize()

    if ckptm.latest_checkpoint:
        print(f'Loaded checkpoint: {ckptm.latest_checkpoint}')
    else:
        print('Initialized from scratch')

    for _ in range(args.iters):
        initial = time.time()
        stargan.train(batch_ds, ckpt, batch_size)

        # show inferred images on tensorflow
        for img, label in dataset.batch(10).take(1):
            label = label2onehot_C(tf.reverse(label, [-1]))

            inferred = stargan.generator([img, label])
            print(inferred.shape)

            with summarywriter.as_default():
                tf.summary.image('image', inferred * 0.5 +
                                 0.5, ckpt.step, max_outputs=10)

        timetaken = time.time() - initial
        logger.info(
            f'Speed: {round(timetaken, 5)} epoch/second, {int(ckpt.step)}')
        ckptm.save()


# @tf.function
# enable tf.function if initial training is successful
def train_step(dataset: tf.data.Dataset,
               generator: keras.Model,
               discriminator: keras.Model,
               encoder: keras.Model,
               mapping: keras.Model,
               summary: tf.summary.SummaryWriter,
               ckpt: tf.train.Checkpoint):

    for image, domain in dataset:
        bs = tf.shape(image)[0]
        ckpt.step.assign_add(1)
        target_d = tf.random.uniform([bs], maxval=2, dtype=tf.int32)

        with tf.GradientTape() as gd:

                # adver_loss(d(x), d(g(x, s))) optimize discriminator
            pass

        with tf.GradientTape() as gt, tf.GradientTape() as gn, tf.GradientTape() as gm:
            # adver_loss(d(x), d(g(x, s))) optimize generator
            pass


def train2(args):
    dir = str(args.dir)
    iters = int(args.iters)
    bs = int(args.bs)
    dataset = create_dataset(dir)

    suffix = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = os.path.join('logdir', suffix, 'train')
    summarywriter = tf.summary.create_file_writer(filename)

    generator = model_v2.Generator()
    discriminator = model_v2.Discriminator()
    encoder = model_v2.Styleencoder()
    mapping = model_v2.Mapping()

    cpkt = tf.train.Checkpoint(step=tf.Variable(1, dtyle=tf.int32),
                               generator=generator,
                               discriminator=discriminator,
                               encoder=encoder,
                               mapping=mapping)
    ckptm = tf.train.CheckpointManager(cpkt, 'checkpoint', 3)
    ckptm.restore_or_initialize()

    if ckptm.latest_checkpoint:
        print(f'Loaded checkpoint: {ckptm.latest_checkpoint}')
    else:
        print('Initialized from scratch')

    batched = dataset.batch(bs, drop_remainder=True)

    for _ in range(iters):
        inittime = time.time()
        train_step(batched, generator, discriminator,
                   encoder, mapping, summary, ckpt)
        tmlapse = time.time() - inittime


if __name__ == "__main__":
    logging.basicConfig(filename='log.txt', level=logging.INFO,
                        format='%(asctime)s  %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory of the celeba',
                        required=True)
    parser.add_argument('--iters', help='number of iterations',
                        default=20)
    parser.add_argument('--bs', help='batch size',
                        default=5)
    logger.info('start training')
    train(parser.parse_args())
