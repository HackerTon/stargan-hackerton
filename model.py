import tensorflow as tf
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.ops.gen_nn_ops import l2_loss
from tensorflow.python.ops.math_ops import reduce_mean
# import scheduler
import tensorflow_addons as tfa
# from tensorflow_examples.models.pix2pix import pix2pix


def resi_block(input_layer, k):
    kernel_initializer = tf.keras.initializers.RandomNormal(0.0, 0.02)

    # first block
    d_block = tf.keras.layers.Conv2D(
        k, (3, 3), (1, 1), padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    d_block = tfa.layers.InstanceNormalization()(d_block)
    d_block = tf.keras.layers.Activation('relu')(d_block)

    # second block
    d2_block = tf.keras.layers.Conv2D(
        k, (3, 3), (1, 1), padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(d_block)
    d2_block = tfa.layers.InstanceNormalization()(d_block)
    d2_block = tf.keras.layers.Activation('relu')(d2_block)

    output = tf.keras.layers.Concatenate()([d2_block, input_layer])

    return output


def generator(nlabel):
    kernel_initializer = tf.keras.initializers.RandomNormal(0.0, 0.02)

    img_input = tf.keras.layers.Input(shape=(None, None, 3))
    cond_input = tf.keras.layers.Input((None, None, nlabel))

    combined_input = tf.keras.layers.Concatenate()([img_input, cond_input])

    # c7s1-64
    conv1 = tf.keras.layers.Conv2D(
        64, (7, 7), 1, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(combined_input)
    conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)

    # d128
    conv2 = tf.keras.layers.Conv2D(
        128, (3, 3), 2, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(conv1)
    conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)

    # d256
    conv3 = tf.keras.layers.Conv2D(
        256, (3, 3), 2, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(conv2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)

    # R256 x 9 times
    res = conv3
    for _ in range(9):
        res = resi_block(res, 256)

    # u128
    deconv1 = tf.keras.layers.Conv2DTranspose(
        128, (3, 3), 2, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(res)
    deconv1 = tfa.layers.InstanceNormalization()(deconv1)
    deconv1 = tf.keras.layers.Activation('relu')(deconv1)

    # u64
    deconv2 = tf.keras.layers.Conv2DTranspose(
        64, (3, 3), 2, padding='same',
        kernel_initializer=kernel_initializer,
        use_bias=False)(deconv1)
    deconv2 = tfa.layers.InstanceNormalization()(deconv2)
    deconv2 = tf.keras.layers.Activation('relu')(deconv2)

    # c7s1-3
    conv4 = tf.keras.layers.Conv2D(
        3, (7, 7), 1, padding='same', use_bias=False)(deconv2)
    conv4 = tfa.layers.InstanceNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('tanh')(conv4)

    return tf.keras.Model(inputs=[img_input, cond_input], outputs=[conv4])


def discriminator(width, nlabel):
    input_layer = tf.keras.layers.Input((width, width, 3))
    conv1 = tf.keras.layers.Conv2D(64, (4, 4),
                                   (2, 2), padding='same')(input_layer)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)

    conv2 = tf.keras.layers.Conv2D(128, (4, 4),
                                   (2, 2), padding='same')(conv1)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)

    conv3 = tf.keras.layers.Conv2D(256, (4, 4),
                                   (2, 2), padding='same')(conv2)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)

    conv4 = tf.keras.layers.Conv2D(512, (4, 4),
                                   (2, 2), padding='same')(conv3)
    conv4 = tf.keras.layers.LeakyReLU()(conv4)

    conv5 = tf.keras.layers.Conv2D(1024, (4, 4),
                                   (2, 2), padding='same')(conv4)
    conv5 = tf.keras.layers.LeakyReLU()(conv5)

    conv6 = tf.keras.layers.Conv2D(2048, (4, 4),
                                   (2, 2), padding='same')(conv5)
    conv6 = tf.keras.layers.LeakyReLU()(conv6)

    src_output = tf.keras.layers.Conv2D(1, (3, 3),
                                        (1, 1), padding='same',
                                        use_bias=False)(conv6)

    cls_output = tf.keras.layers.Conv2D(nlabel, (width // 64, width // 64),
                                        (1, 1), padding='valid',
                                        use_bias=False)(conv6)

    return tf.keras.Model(inputs=[input_layer], outputs=[src_output, cls_output])


def label2onehot(label, width, nlabels):
    batch_size = label.shape[0]

    indices = tf.constant(label)
    shape = tf.constant([batch_size, width, width, nlabels])
    onehot = tf.scatter_nd()


def adverserial_loss(domain, target):
    adv_1 = tf.reduce_mean(tf.math.log(tf.sigmoid(domain)))
    adv_2 = tf.reduce_mean(tf.math.log(1 - tf.sigmoid(target)))

    return adv_1 + adv_2


def reconstruction_loss(realimg, rerealimg):
    return tf.reduce_mean(tf.abs(realimg - rerealimg))


class Stargan:
    def __init__(self, summary, width, nlabel, lambda_cls=1, lambda_rec=10, lambda_gp=10):
        self.discriminator = discriminator(width, nlabel)
        self.generator = generator(nlabel)
        self.disopti = tf.keras.optimizers.Adam(1e-4, 0.5)
        self.genopti = tf.keras.optimizers.Adam(1e-4, 0.5)
        self.summary: tf.summary.SummaryWriter = summary
        self.width = width
        self.binloss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.nlabel = nlabel

        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_gp = lambda_gp

    @tf.function
    def train_step(self, domimg, domcond, step):
        batch_size = domimg.shape[0]

        random = tf.random.uniform([batch_size], maxval=2, dtype=tf.int32)
        tarcond = tf.one_hot(random, 2)

        targetcond = tf.TensorArray(tf.float32, batch_size, dynamic_size=True)
        domaincond = tf.TensorArray(tf.float32, batch_size, dynamic_size=True)

        # Convert label into onehot
        for i in tf.range(0, batch_size):
            mask1 = tarcond[i, 0]
            mask2 = tarcond[i, 1]
            mask11 = domcond[i, 0]
            mask22 = domcond[i, 1]

            tarcombined = tf.concat([tf.fill((self.width, self.width, 1), mask1),
                                     tf.fill((self.width, self.width, 1), mask2)], -1)
            srccombined = tf.concat([tf.fill((self.width, self.width, 1), mask11),
                                     tf.fill((self.width, self.width, 1), mask22)], -1)

            targetcond.write(i, tarcombined)
            domaincond.write(i, srccombined)

        with tf.GradientTape(persistent=True) as tape:
            domresc, domcls = self.discriminator(domimg)
            classreal = self.binloss(domcond, tf.reshape(domcls,
                                                         (batch_size, self.nlabel)))
            faketarimg = self.generator([domimg, targetcond.stack()])
            tarresc, tarcls = self.discriminator(faketarimg)

            real = -tf.reduce_mean(domresc)
            fake = tf.reduce_mean(tarresc)

            eps = tf.random.uniform([batch_size, 1, 1, 1], 0, 1)

            with tf.GradientTape() as gp_tape:
                x_hat = eps * domimg + (1.0 - eps) * faketarimg
                gp_tape.watch(x_hat)

                xhatresc, _ = self.discriminator(x_hat)

            grad = gp_tape.gradient(xhatresc, x_hat)

            l2_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), [1, 2, 3]))
            grad_penal = tf.reduce_mean(tf.square(l2_norm - 1.0))

            totaldisloss = real + fake + self.lambda_cls * classreal
            totaldisloss += self.lambda_gp * grad_penal

        with self.summary.as_default():
            tf.summary.scalar('disreal', real, step)
            tf.summary.scalar('disfake', fake, step)
            tf.summary.scalar('gradpenal', grad_penal, step)

        grads = tape.gradient(totaldisloss,
                              self.discriminator.trainable_variables)
        self.disopti.apply_gradients(
            zip(grads, self.discriminator.trainable_variables))

        if step % 5 == 0:
            with tf.GradientTape() as tape:
                faketarimg = self.generator([domimg, targetcond.stack()])
                fakedomimg = self.generator([faketarimg, domaincond.stack()])
                tarresc, tarcls = self.discriminator(faketarimg)

                lossadv = -tf.reduce_mean(tarresc)
                losscls = self.binloss(tarcond, tf.reshape(tarcls,
                                                           (batch_size, self.nlabel)))
                lossrec = reconstruction_loss(domimg, fakedomimg)

                totalganloss = lossadv + self.lambda_cls * losscls
                totalganloss += self.lambda_rec * lossrec

            grads = tape.gradient(totalganloss,
                                  self.generator.trainable_variables)
            self.genopti.apply_gradients(zip(grads,
                                             self.generator.trainable_variables))

            with self.summary.as_default():
                tf.summary.scalar('ganlossadv', lossadv, step)
                tf.summary.scalar('losscls', lossadv, step)
                tf.summary.scalar('lossrec', lossrec, step)
