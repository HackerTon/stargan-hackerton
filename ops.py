import os

import numpy as np
import tensorflow as tf



def readdecode(filename, attr):
    raw = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[20:-20, :]
    image = tf.image.resize(image, (128, 128))
    image = image * 2 - 1

    attr = [0.0, 1.0] if attr == b'1' else [1.0, 0.0]

    return image, attr


def readdecode2(filename, attr):
    raw = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[20:-20, :]
    image = tf.image.resize(image, (128, 128))
    image = image * 2 - 1

    return image, attr


def textparser(text):
    strings = tf.strings.split(text, ' ')
    mask = tf.strings.regex_full_match(strings, '-?1')
    new_strings = tf.boolean_mask(strings, mask)
    link = strings[0]
    domain = tf.strings.to_number(new_strings)
    domain = domain * 0.5 + 0.5
    domain = domain[-20]

    return link, domain


def label2onehot_C(label):
    batch_size = tf.shape(label)[0]
    arr = np.zeros([batch_size, 128, 128, 2])

    for i in range(batch_size):
        arr[i, :, :, 0] = label[i, 0]
        arr[i, :, :, 1] = label[i, 1]

    return arr


def create_dataset_celb(dir):
    filepath = os.path.join(dir, 'list_attr_celeba.txt')
    imgdir = os.path.join(dir, 'img_align_celeba')

    textfile = tf.data.TextLineDataset(filepath)
    textfile = textfile.map(textparser)

    adddir = lambda x, y: (imgdir + '/' + x, y)
    link2image = lambda link, attr: readdecode(link, attr)

    ds = textfile.map(adddir)
    ds = ds.map(link2image)

    return ds


def create_dataset(dir):
    filepath = os.path.join(dir, 'list_attr_celeba.txt')
    imgdir = os.path.join(dir, 'img_align_celeba')

    textfile = tf.data.TextLineDataset(filepath)
    textfile = textfile.map(textparser)

    adddir = lambda x, y: (imgdir + '/' + x, y)
    link2image = lambda link, attr: readdecode2(link, attr)

    ds = textfile.map(adddir)
    ds = ds.map(link2image)

    return ds
