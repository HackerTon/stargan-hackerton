import tensorflow as tf
import time


def benchmark_dataset(dataset):

    itime = time.time()
    for datum in dataset:
        tf.reduce_mean(datum[0])
    timetaken = time.time() - itime

    print(f'Time taken: {round(timetaken, 7)}s')
