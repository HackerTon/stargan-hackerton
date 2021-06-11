import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from functools import partial

from model_v2 import InstanceNorm


def reconstruction_loss(realimg, rerealimg):
    return tf.reduce_mean(tf.abs(realimg - rerealimg))


def one2hot(attr, nlabel, shape=[128, 128]):
    indices = tf.where(tf.equal(attr, 1))
    updates = tf.ones([tf.shape(indices)[0], shape[0], shape[1]])

    cube = tf.scatter_nd(
        indices, updates, [nlabel, shape[0], shape[1]]
    )  # [nlabel, shape[0], shape[1]]
    cube = tf.transpose(cube, [1, 2, 0])  # [shape[0], shape[1], nlabel]

    return cube


def resi_block(input_layer, k):
    # first block
    d_block = tf.keras.layers.Conv2D(k, (3, 3), (1, 1), padding="same")(input_layer)
    d_block = InstanceNorm()(d_block)
    d_block = tf.keras.layers.ReLU()(d_block)

    # second block
    d2_block = tf.keras.layers.Conv2D(k, (3, 3), (1, 1), padding="same")(d_block)
    d2_block = InstanceNorm()(d2_block)
    d2_block = tf.keras.layers.ReLU()(d2_block)

    output = tf.keras.layers.Add()([d2_block, input_layer])

    return output


def generator(nlabel):
    img_input = tf.keras.layers.Input(shape=(128, 128, 3))
    cond_input = tf.keras.layers.Input((128, 128, nlabel))

    combined_input = tf.keras.layers.Concatenate()([img_input, cond_input])

    # c7s1-64
    conv1 = tf.keras.layers.Conv2D(64, (7, 7), 1, padding="same")(combined_input)
    conv1 = InstanceNorm()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)

    # d128
    conv2 = tf.keras.layers.Conv2D(128, (4, 4), 2, padding="same")(conv1)
    conv2 = InstanceNorm()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)

    # d256
    conv3 = tf.keras.layers.Conv2D(256, (4, 4), 2, padding="same")(conv2)
    conv3 = InstanceNorm()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)

    # R256 x 9 times
    res = conv3
    for _ in range(9):
        res = resi_block(res, 256)

    # u128
    deconv1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), 2, padding="same")(res)
    deconv1 = InstanceNorm()(deconv1)
    deconv1 = tf.keras.layers.ReLU()(deconv1)

    # u64
    deconv2 = tf.keras.layers.Conv2DTranspose(64, (3, 3), 2, padding="same")(deconv1)
    deconv2 = InstanceNorm()(deconv2)
    deconv2 = tf.keras.layers.ReLU()(deconv2)

    # c7s1-3
    conv4 = tf.keras.layers.Conv2D(3, (7, 7), 1, padding="same")(deconv2)
    conv4 = InstanceNorm()(conv4)
    conv4 = tf.keras.layers.Activation("tanh")(conv4)

    return tf.keras.Model(
        inputs=[img_input, cond_input], outputs=[conv4], name="generator"
    )


def discriminator(width, nlabel):
    input_layer = tf.keras.layers.Input((128, 128, 3))
    output = input_layer

    for i in range(6):
        output = tf.keras.layers.Conv2D((2 ** i) * 64, (4, 4), (2, 2), padding="same")(
            output
        )
        output = tf.keras.layers.LeakyReLU(0.01)(output)

    src_output = tf.keras.layers.Conv2D(1, (3, 3), (1, 1), padding="same", name="src")(
        output
    )

    cls_output = tf.keras.layers.Conv2D(
        nlabel,
        (width // 64, width // 64),
        (1, 1),
        padding="valid",
        name="cls",
    )(output)

    return tf.keras.Model(
        inputs=[input_layer], outputs=[src_output, cls_output], name="discriminator"
    )

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr):
        self.init_lr = init_lr
    
    def __call__(self, step):
        return self.init_lr / (step + 1)

class Stargan:
    def __init__(
        self,
        summary,
        width,
        nlabel,
        lambda_cls=1,
        lambda_rec=10,
        lambda_gp=10,
        scheduler=False,
    ):
        self.discriminator = discriminator(width, nlabel)
        self.generator = generator(nlabel)

        if scheduler:
            schedule = 


            self.disopti = tf.keras.optimizers.Adam(1e-4, 0.5)
            self.genopti = tf.keras.optimizers.Adam(1e-4, 0.5)
        else:
            self.disopti = tf.keras.optimizers.Adam(1e-4, 0.5)
            self.genopti = tf.keras.optimizers.Adam(1e-4, 0.5)

        self.summary: tf.summary.SummaryWriter = summary
        self.binloss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.width = width
        self.nlabel = nlabel
        self.lambda_cls = float(lambda_cls)
        self.lambda_rec = float(lambda_rec)
        self.lambda_gp = float(lambda_gp)

    # @tf.function
    def train(self, dataset, checkpoint, batch_size):
        for img, label in dataset:
            self.train_step(img, label, checkpoint.step, batch_size)
            checkpoint.step.assign_add(1)

    def train_step(self, domimg, domcond, step, batch_size):
        rand_idx = tf.random.uniform(
            tf.expand_dims(batch_size, 0), dtype=tf.int32, maxval=2
        )

        # tarcond, domcond (batchsize, number label)
        # must be a binary attributes, not softmax
        # [batchsize][1, 1, 0, 1, ...]
        tarcond = tf.random.uniform([batch_size, self.nlabel], maxval=2, dtype=tf.int32)

        one2hot_p = partial(one2hot, nlabel=self.nlabel)
        targetcond = tf.map_fn(one2hot_p, elems=tarcond, dtype=tf.float32)
        domaincond = tf.map_fn(one2hot_p, elems=domcond, dtype=tf.float32)

        # backpropagation for discriminator
        with tf.GradientTape(persistent=True) as tape:
            domresc, domcls = self.discriminator(domimg)

            classreal = self.binloss(
                domcond, tf.reshape(domcls, (batch_size, self.nlabel))
            )
            faketarimg = self.generator([domimg, targetcond])
            tarresc, tarcls = self.discriminator(faketarimg)

            real = -tf.reduce_mean(domresc)
            fake = tf.reduce_mean(tarresc)
            eps = tf.random.uniform([batch_size, 1, 1, 1], 0, 1)

            with tf.GradientTape() as gp_tape:
                x_hat = eps * domimg + (1.0 - eps) * faketarimg
                gp_tape.watch(x_hat)
                xhatresc, _ = self.discriminator(x_hat)

            grad = gp_tape.gradient(xhatresc, x_hat)

            # this is correctly implemented
            l2_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), [1, 2, 3]))
            grad_penal = tf.reduce_mean(tf.square(l2_norm - 1.0))

            totaldisloss = real + fake + self.lambda_cls * classreal
            totaldisloss += self.lambda_gp * grad_penal

        with self.summary.as_default():
            tf.summary.scalar("disreal", real, step)
            tf.summary.scalar("disfake", fake, step)
            tf.summary.scalar("discls", classreal, step)
            tf.summary.scalar("gp", grad_penal, step)

        grads = tape.gradient(totaldisloss, self.discriminator.trainable_variables)
        self.disopti.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # backpropagation for generator
        if (step + 1) % 5 == 0:
            with tf.GradientTape() as tape:
                faketarimg = self.generator([domimg, targetcond])
                fakedomimg = self.generator([faketarimg, domaincond])
                tarresc, tarcls = self.discriminator(faketarimg)

                lossadv = -tf.reduce_mean(tarresc)
                losscls = self.binloss(
                    tarcond, tf.reshape(tarcls, (batch_size, self.nlabel))
                )
                lossrec = reconstruction_loss(domimg, fakedomimg)

                totalganloss = lossadv + self.lambda_cls * losscls
                totalganloss += self.lambda_rec * lossrec

            grads = tape.gradient(totalganloss, self.generator.trainable_variables)
            self.genopti.apply_gradients(zip(grads, self.generator.trainable_variables))

            with self.summary.as_default():
                tf.summary.scalar("ganlossadv", lossadv, step)
                tf.summary.scalar("losscls", lossadv, step)
                tf.summary.scalar("lossrec", lossrec, step)
