import tensorflow as tf
import tensorflow_addons as tfa
from non_local import sn_non_local_block

class usample(tf.keras.layers.Layer):
    def __init__(self, factor=2, name="usample"):
        super(usample, self).__init__(name=name)
        self.factors = factor

    def call(self, x):
        _, nh, nw, _ = x.shape
        x = tf.image.resize(x, [nh*self.factors, nw*self.factors])
        return x

class block(tf.keras.layers.Layer):
    def __init__(self, out_channels, name="block"):
        super(block, self).__init__(name=name)
        self.out_channels = out_channels

        # sub layers
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.snconv2d_1 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, strides=1, padding='same')
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.snconv2d_2 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, strides=1, padding='same')
        )
        self.snconv2d_3 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=1, strides=1, padding='same')
        )

        self.usample = usample()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=True, usample=True):
        x_0 = x
        x = self.bn0(x, training)
        x = self.relu(x)
        if usample:
            x = self.usample(x)
        x = self.snconv2d_1(x, training=training)
        x = self.bn1(x, training)
        x = self.relu(x)
        x = self.snconv2d_2(x, training=training)

        if usample:
            x_0 = self.usample(x_0)
        x_0 = self.snconv2d_3(x_0, training=training)
        return x_0 + x

class generator(tf.keras.Model):
    def __init__(self, gf_dim, name="generator"):
        super(generator, self).__init__(name=name)
        self.gf_dim = gf_dim

        # sub layers
        self.snlinear = tfa.layers.SpectralNormalization(
            tf.keras.layers.Dense(self.gf_dim * 8 * 7 * 7)
        )
        self.block0 = block(self.gf_dim * 8) # 14 * 14
        self.block1 = block(self.gf_dim * 4) # 28 * 28
        self.sn_non_local_block = sn_non_local_block(self.gf_dim*4)
        self.block2 = block(self.gf_dim * 2)  # 28 * 28
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.snconv1 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same')
        )
        self.relu = tf.keras.layers.ReLU()
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, x, training=True):
        x = self.snlinear(x, training)
        x = tf.reshape(x, [-1, 7, 7, self.gf_dim * 8])
        x = self.block0(x, training)
        x = self.block1(x, training)
        x = self.sn_non_local_block(x)
        x = self.block2(x, training, usample=False)
        x = self.bn0(x, training)
        x = self.relu(x)
        x = self.snconv1(x, training)
        x = self.tanh(x)

        return x
