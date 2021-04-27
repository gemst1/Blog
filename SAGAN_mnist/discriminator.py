import tensorflow as tf
import tensorflow_addons as tfa
from non_local import sn_non_local_block

class dsample(tf.keras.layers.Layer):
    def __init__(self, factors=2, name="dsample"):
        super(dsample, self).__init__(name)
        self.factors = factors
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=self.factors)

    def call(self, x):
        x = self.avgpool(x)
        return x

class optimized_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, name="opt_blcok"):
        super(optimized_block, self).__init__(name=name)
        self.out_channels = out_channels

        # sub layers
        self.snconv2d_1 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, strides=1, padding="same")
        )
        self.snconv2d_2 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, strides=1, padding="same")
        )
        self.snconv2d_3 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=1, strides=1, padding="same")
        )
        self.dsample = dsample()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, downsample=True, sn_training=True):
        x_0 = x
        x = self.snconv2d_1(x, sn_training)
        x = self.relu(x)
        x = self.snconv2d_2(x, sn_training)
        if downsample:
            x = dsample(x)
            x_0 = dsample(x_0)
        x_0 = self.snconv2d_3(x_0, sn_training)

        return x_0 + x

class block(tf.keras.layers.Layer):
    def __init__(self, out_channels, name="block"):
        super(block, self).__init__(name=name)
        self.out_channels = out_channels

        # sub layers
        self.snconv2d_1 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, strides=1, padding="same")
        )
        self.snconv2d_2 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, strides=1, padding="same")
        )
        self.snconv2d_3 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=1, strides=1, padding="same")
        )
        self.dsample = dsample()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, downsample= True, sn_training=True):
        input_channels = x.shape[-1]
        x_0 = x
        x = self.relu(x)
        x = self.snconv2d_1(x, sn_training)
        x = self.relu(x)
        x = self.snconv2d_2(x, sn_training)
        if downsample:
            x = self.dsample(x)
        if downsample or input_channels != self.out_channels:
            x_0 = self.snconv2d_3(x_0, sn_training)
            if downsample:
                x_0 = self.dsample(x_0)

        return x_0 + x

class discriminator(tf.keras.Model):
    def __init__(self, df_dim, name="discriminator"):
        super(discriminator, self).__init__(name=name)
        self.df_dim = df_dim

        # sub layers
        self.optimized_block = optimized_block(self.df_dim)
        self.block0 = block(self.df_dim*2)
        self.sn_non_local_block = sn_non_local_block(self.df_dim*2)
        self.block1 = block(self.df_dim*4)
        self.block2 = block(self.df_dim*8)
        self.block3 = block(self.df_dim*8)
        self.relu = tf.keras.layers.ReLU()
        self.snlinear = tfa.layers.SpectralNormalization(
            tf.keras.layers.Dense(1)
        )

    def call(self, x, sn_training=True):
        x = self.optimized_block(x, downsample=False, sn_training=sn_training) # 28 * 28
        x = self.block0(x, downsample=False, sn_training=sn_training) # 28 * 28
        x = self.sn_non_local_block(x, sn_training) # 28 * 28
        x = self.block1(x, sn_training=sn_training) # 14 * 14
        x = self.block2(x, sn_training=sn_training) # 7 * 7
        x = self.block3(x, downsample=False, sn_training=sn_training) # 7 * 7
        x = self.relu(x)
        x = tf.reduce_sum(x, [1, 2])
        x = self.snlinear(x, sn_training)

        return x