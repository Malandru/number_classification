import tensorflow as tf
import numpy as np
import cv2

HEIGHT = 512
WIDTH = 512


def load_number_dataset(length=100):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Reduce dataset to <length>
    x_train = x_train[:length]
    y_train = y_train[:length]
    x_test = x_test[:length]
    y_test = y_test[:length]

    train = transform_dataset(x_train, y_train)
    test = transform_dataset(x_test, y_test)
    return train, test


def transform_dataset(x, y):
    new_x_shape = (x.shape[0], HEIGHT, WIDTH, 3)
    new_y_shape = (y.shape[0], HEIGHT, WIDTH, 1)
    new_x = np.zeros(shape=new_x_shape)
    new_y = np.zeros(shape=new_y_shape)

    for i in range(len(x)):
        new_x[i] = create_three_channel_image(x[i])
        new_y[i] = create_mask_image(new_x[i])
    return new_x, new_y


def create_three_channel_image(np_array):
    w, h = np_array.shape
    img = np.zeros((w, h, 3))
    for i in range(w):
        for j in range(h):
            if np_array[(i, j)] <= 0:
                img[(i, j, 0)] = 255
                img[(i, j, 1)] = 255
                img[(i, j, 2)] = 255
    return cv2.resize(img, dsize=(HEIGHT, WIDTH))


def create_mask_image(np_array):
    w, h, _ = np_array.shape
    img = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if np_array[(i, j, 0)] > 0:
                img[(i, j)] = 255
    img = cv2.resize(img, dsize=(HEIGHT, WIDTH))
    return np.expand_dims(img, axis=-1)
