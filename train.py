import argparse
import datetime
import logging
import os
import time

import numpy as np
import tensorflow as tf

import model

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.random.set_seed(1234567)


def readdecode(filename, attr):
    raw = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[20:-20, :]
    image = tf.image.resize(image, (128, 128))
    image = image * 2 - 1

    attr = [0.0, 1.0] if attr == b'1' else [1.0, 0.0]

    return image, attr


def textparser(text):
    strings = tf.strings.split(text, ' ')
    mask = tf.strings.regex_full_match(strings, '-?1')
    new_strings = tf.boolean_mask(strings, mask)

    link = strings[0]

    return link, new_strings[-20]


def create_dataset_celb(dir):
    filepath = os.path.join(dir, 'list_attr_celeba.txt')
    imgdir = os.path.join(dir, 'img_align_celeba')

    textfile = tf.data.TextLineDataset(filepath, num_parallel_reads=AUTOTUNE)
    textfile = textfile.map(textparser, AUTOTUNE)

    adddir = lambda x, y: (imgdir + '/' + x, y)
    link2image = lambda link, attr: readdecode(link, attr)

    ds = textfile.map(adddir, AUTOTUNE)
    ds = ds.map(link2image, AUTOTUNE)

    return ds


def label2onehot_C(label):
    batch_size = label.shape[0]
    arr = np.zeros([batch_size, 128, 128, 2])

    for i in range(batch_size):
        arr[i, :, :, 0] = label[i, 0]
        arr[i, :, :, 1] = label[i, 1]

    return arr


def train(args):
    batch_size = args.bs

    logger = logging.getLogger(__name__)
    dataset = create_dataset_celb(args.dir)
    batch_ds = dataset.shuffle(10000).batch(batch_size)

    suffix = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = os.path.join('logdir', suffix, 'train')
    summarywriter = tf.summary.create_file_writer(filename)

    stargan = model.Stargan(summarywriter, 128, 2)
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1, dtype=tf.int64),
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

        for img, label in dataset.batch(10).take(1):
            label = label2onehot_C(tf.reverse(label, [-1]))

            inferred = stargan.generator([img, label])

            with summarywriter.as_default():
                tf.summary.image('image', inferred * 0.5 + 0.5, ckpt.step)

        timetaken = time.time() - initial
        logger.info(
            f'Speed: {round(timetaken, 5)} epoch/second, {int(ckpt.step)}')
        ckptm.save()


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
