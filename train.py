import os
import argparse

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def readdecode(filename, width):
    """
    Can only read JPEG type of file
    """
    raw = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (width, width))
    # image = tf.image.random_crop(image, [WIDTH, WIDTH, 3])
    image = image * 2 - 1

    return image


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

    fmale = textfile.filter(lambda _, gender: gender == '-1')
    male = textfile.filter(lambda _, gender: gender == '1')

    adddir = lambda x, y: (dir + 'img_align_celeba/' + x, y)

    fmale = fmale.map(adddir, AUTOTUNE)
    male = male.map(adddir, AUTOTUNE)

    link2image = lambda link, gender: readdecode(link, width)

    fmale = fmale.map(link2image, AUTOTUNE)
    male = male.map(link2image, AUTOTUNE)

    return fmale, male

def train(args):
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    train(args)
    
    