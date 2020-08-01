import os
import argparse
import model
import datetime

import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


def readdecode(filename, attr, width):
    raw = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (width, width))
    image = image * 2 - 1

    attr = [0.0, 1.0] if attr == b'1' else [0.0, 1.0]

    return image, attr


def textparser(text):
    strings = tf.strings.split(text, ' ')
    mask = tf.strings.regex_full_match(strings, '-?1')
    new_strings = tf.boolean_mask(strings, mask)

    link = strings[0]

    return link, new_strings[-20]


def create_dataset_celb(dir, width):
    filepath = os.path.join(dir, 'list_attr_celeba.txt')

    textfile = tf.data.TextLineDataset(filepath)
    textfile = textfile.map(textparser)

    adddir = lambda x, y: (dir + 'img_align_celeba/' + x, y)
    link2image = lambda link, attr: readdecode(link, attr, width)

    ds = textfile.map(adddir)
    ds = ds.map(link2image)

    return ds


def label2onehot_C(label):
    arr = np.zeros([1, 128, 128, 2])

    arr[:, :, :, 0] = label[0]
    arr[:, :, :, 1] = label[1]

    return arr


def train(args):
    dataset = create_dataset_celb(args.dir, 128)
    batch_ds = dataset.take(10000).shuffle(10000).batch(5)

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
        print(f'Loaded checkpoint: {ckptm.latest_checkpoint}')
    ckptm.restore_or_initialize()

    step = 0
    for it in range(args.iters):
        for img, label in batch_ds:
            step += 1
            stargan.train_step(img, label, tf.constant(step, tf.int64))

        for img in dataset.take(1):
            label = label2onehot_C([0, 1])
            infered = stargan.generator([tf.expand_dims(img[0], 0), label])

            with summarywriter.as_default():
                tf.summary.image('image', infered, step)

        if it % 5 == 0:
            ckptm.save(it)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory of the celeba',
                        required=True)
    parser.add_argument('--iters', help='number of iterations',
                        default=20)

    train(parser.parse_args())
