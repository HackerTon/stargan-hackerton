import os
import tensorflow as tf
from model import generator as gen

# Disable GPU initialization
os.environ['CUDA_VISIBLE_DEVICES'] = -1


def main(*args, **kargs):
    c_generator: tf.keras.Model = gen(2)

    checkpt = tf.train.Checkpoint(generator=c_generator)
    checkptm = tf.train.CheckpointManager(checkpt, 'checkpoint', 3)

    if checkptm.latest_checkpoint is None:
        print('No checkpoint available in checkpoint/')
        return

    print('List of checkpoints:')
    print('Choose your desired checkpoint')
    checkpoints = checkptm.checkpoints

    for num, checkpoint in enumerate(checkpoints):
        print(f'{num}: {checkpoint}')

    selection = int(input())

    if selection in range(len(checkpoints)):
        checkpt.restore(checkpoints[selection])

        c_generator.save('inferencing', include_optimizer=False)
    else:
        print('You have selected checkpoint that is not in the range.')
        return


if __name__ == '__main__':
    main()
