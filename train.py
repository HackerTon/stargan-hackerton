import argparse
import datetime
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

    textfile = tf.data.TextLineDataset(filepath)
    textfile = textfile.map(textparser, AUTOTUNE)

    adddir = lambda x, y: (imgdir + '/' + x, y)
    link2image = lambda link, attr: readdecode(link, attr)

    ds = textfile.map(adddir, AUTOTUNE)
    ds = ds.map(link2image, AUTOTUNE)

    return ds


ds = create_dataset_celb('/home/hackerton/cyclegan_dataset')


def label2onehot_C(label):
    batch_size = label.shape[0]
    arr = np.zeros([batch_size, 128, 128, 2])

    for i in range(batch_size):
        arr[i, :, :, 0] = label[i, 0]
        arr[i, :, :, 1] = label[i, 1]

    return arr


def train(args):
    dataset = create_dataset_celb(args.dir)
    batch_ds = dataset.take(1000).shuffle(10000).batch(5)

    suffix = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = os.path.join('logdir', suffix, 'train')
    summarywriter = tf.summary.create_file_writer(filename)

    stargan = model.Stargan(summarywriter, 128, 2)
    ckpt = tf.train.Checkpoint(
        generator=stargan.generator,
        discrimintor=stargan.discriminator,
        genopti=stargan.genopti,
        disopti=stargan.disopti
    )

    ckptm = tf.train.CheckpointManager(ckpt, 'checkpoint', 10)
    if ckptm.latest_checkpoint:
        print(tf.train.list_variables(ckptm.latest_checkpoint))
        print(f'Loaded checkpoint: {ckptm.latest_checkpoint}')
    ckptm.restore_or_initialize()

    step = 0

    if os.path.isfile('iter.log'):
        step_log = open('iter.log', 'r+')
        readout = step_log.read()

        if readout != '':
            step = int(readout)
    else:
        step_log = open('iter.log', 'w+')

    for it in range(args.iters):
        initial = time.time()

        for img, label in batch_ds:
            stargan.train_step(img, label, tf.constant(step, tf.int64))
            step += 1

        for img, label in dataset.batch(2).take(1):
            label = label2onehot_C(tf.reverse(label, [-1]))

            inferred = stargan.generator([img, label])

            with summarywriter.as_default():
                tf.summary.image('image', inferred * 0.5 + 0.5, step)

        timetaken = time.time() - initial

        print(f'Timetaken per epoch: {round(timetaken, 5)}, {step}')
        step_log.seek(0)
        step_log.write(repr(step))

        if it % 1 == 0:
            ckptm.save(it)

    step_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory of the celeba',
                        required=True)
    parser.add_argument('--iters', help='number of iterations',
                        default=20)

    train(parser.parse_args())
