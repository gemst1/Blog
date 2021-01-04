from datetime import datetime
import io

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

logdir = './logs/image/train_data/' + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

def plot_to_image(figure):
    # convert the matplotlib figure to png image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # add bath dimension
    image = tf.expand_dims(image, 0)
    return image

def image_grid():
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1, title=class_names[train_labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)

    return figure

figure = image_grid()
with file_writer.as_default():
    tf.summary.image("Training data", plot_to_image(figure), step=0)
