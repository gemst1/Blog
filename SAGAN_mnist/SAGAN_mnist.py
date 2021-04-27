import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from discriminator import discriminator
from generator import generator

class SAGAN_mnist():
    def __init__(self,
                 batch_size_per_replica=128,
                 epochs=20,
                 gf_dim=32,
                 df_dim=32):

        self.batch_size_per_replica = batch_size_per_replica
        self.epochs = epochs
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.latent_dim = 128

        # load mnist data
        self.data = self.load_data()
        self.buffer_size = len(self.data)

        # distributed strategy
        self.strategy = tf.distribute.MirroredStrategy()
        self.replicas = self.strategy.num_replicas_in_sync
        self.batch_size = self.batch_size_per_replica * self.replicas
        print("Number of Devices: {}".format(self.replicas))

        self.dataset = tf.data.Dataset.from_tensor_slices(self.data).shuffle(self.buffer_size)\
            .batch(self.batch_size, drop_remainder=True)
        self.dist_dataset = self.strategy.experimental_distribute_dataset(self.dataset)

        with self.strategy.scope():
            # Model
            self.create_model()

            # Optimizer
            self.g_optimizer = tf.keras.optimizers.Adam(0.0001, 0.0, 0.9)
            self.d_optimizer = tf.keras.optimizers.Adam(0.0002, 0.0, 0.9)

    def load_data(self):
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        data = np.concatenate([x_train, x_test])
        data = data.astype("float32") / 255.0
        data = np.reshape(data, (-1, 28, 28, 1))
        return data

    def create_model(self):
        self.generator = generator(self.gf_dim)
        self.discriminator = discriminator(self.df_dim)

    # Loss functions
    def generator_loss(self, discriminator_on_generator):
        return -tf.reduce_sum(discriminator_on_generator) * (1. / self.batch_size)

    def disc_real_loss(self, discriminator_on_data):
        loss = tf.nn.relu(1.0 - discriminator_on_data)
        return tf.reduce_sum(loss) * (1. / self.batch_size)

    def disc_fake_loss(self, discriminator_on_generator):
        return tf.reduce_sum(tf.nn.relu(1 + discriminator_on_generator)) * (1. / self.batch_size)

    def train_step(self, real_img):
        z = tf.random.normal(shape=(self.batch_size_per_replica, self.latent_dim))
        fake_img = self.generator(z)

        with tf.GradientTape() as tape:
            discriminator_on_data = self.discriminator(real_img)
            discriminator_on_generator = self.discriminator(fake_img, sn_training=False)
            d_loss_real = self.disc_real_loss(discriminator_on_data)
            d_loss_fake = self.disc_fake_loss(discriminator_on_generator)
            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        z = tf.random.normal(shape=(self.batch_size_per_replica, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_img = self.generator(z)
            discriminator_on_generator = self.discriminator(fake_img, sn_training=False)
            g_loss = self.generator_loss(discriminator_on_generator)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return d_loss_real, d_loss_fake, g_loss, fake_img

    @tf.function
    def distributed_train_step(self, dataset):
        per_d_loss_real, per_d_loss_fake, per_g_loss, gen_image = self.strategy.run(self.train_step, args=(dataset,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_d_loss_real, axis=None), \
               self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_d_loss_fake, axis=None), \
               self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_g_loss, axis=None), \
               gen_image

    def train(self):
        # train loop
        for epoch in range(self.epochs):
            print("\nStrat epoch %d" % epoch)
            d_loss_total = 0.0
            g_loss_total = 0.0
            num_batches = 0
            for step, real_img_batch in enumerate(self.dist_dataset):
                d_loss_real, d_loss_fake, g_loss, gen_img = self.distributed_train_step(real_img_batch)
                num_batches += 1
                d_loss_total += d_loss_real + d_loss_fake
                g_loss_total += g_loss

                if step % 50 == 0:
                    # Print metrics
                    print("discriminator loss at step %d: %.2f" % (step, d_loss_real+d_loss_fake))
                    print("\treal loss at step %d: %.2f" % (step, d_loss_real))
                    print("\tfake loss at step %d: %.2f" % (step, d_loss_fake))
                    print("adversarial loss at step %d: %.2f" % (step, g_loss))

                    # Save one generated image
                    img = tf.keras.preprocessing.image.array_to_img(
                        (gen_img.values[0][0]+1.) * 127.5, scale=False
                    )
                    img.save("./gan_results/generated_img" + str(epoch) + "_" + str(step) + ".png")

            if epoch % 10 == 0:
                for i in range(16):
                    img = tf.keras.preprocessing.image.array_to_img(
                        (gen_img.values[0][i]+1) * 127.5, scale=False
                    )
                    img.save("./gan_results/img_" + str(epoch) +"_" + str(i) + ".png")

if __name__ == "__main__":
    SAGAN_mnist = SAGAN_mnist(batch_size_per_replica=128,
                              epochs=100,
                              gf_dim=64,
                              df_dim=64)
    SAGAN_mnist.train()
    # UfO_pusher.evaluation('./results/models/model_80000', 20)