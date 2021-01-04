from datetime import datetime

import tensorflow as tf
from tensorflow import keras

import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

img = np.reshape(train_images[0], (-1, 28, 28, 1))

logdir = './logs/image/train_data/' + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

# single image
# with file_writer.as_default():
#     tf.summary.image('Training data', img, step=0)

# multiple images
with file_writer.as_default():
    images = np.reshape(train_images[:25], (-1, 28, 28, 1))
    tf.summary.image("25 training data examples", images, max_outputs=10, step=0)