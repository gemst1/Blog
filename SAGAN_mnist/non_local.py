import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

class sn_non_local_block(tf.keras.layers.Layer):
    def __init__(self, num_channles, name="sn_non_local"):
        super(sn_non_local_block, self).__init__(name=name)
        self.ch = num_channles

        # sub layers
        self.snconv1 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.ch // 8, kernel_size=1, strides=1, padding='same')
        )
        self.snconv2 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.ch // 8, kernel_size=1, strides=1, padding='same')
        )
        self.snconv3 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.ch // 2, kernel_size=1, strides=1, padding='same')
        )
        self.snconv4 = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.ch, kernel_size=1, strides=1, padding='same')
        )
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.softmax = tf.keras.layers.Softmax()
        self.sigma = self.add_weight("sigma", (), initializer="zeros", trainable=True)


    def call(self, x, sn_training=True):
        batch_size, h, w , channels = x.shape
        location_num = h * w
        dsample_num = location_num // 4

        assert self.ch == channels, "Input's channel size should be same with layer's filter size."

        # theta path
        theta = self.snconv1(x, sn_training)
        theta = tf.reshape(theta, [batch_size, location_num, self.ch // 8])

        # phi path
        phi = self.snconv2(x, sn_training)
        phi = self.maxpool(phi)
        phi = tf.reshape(phi, [batch_size, dsample_num, self.ch // 8])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = self.softmax(attn)

        # g path
        g = self.snconv3(x, sn_training)
        g = self.maxpool(g)
        g = tf.reshape(g, [batch_size, dsample_num, self.ch // 2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, self.ch // 2])
        attn_g = self.snconv4(attn_g, sn_training)

        return x + self.sigma * attn_g
