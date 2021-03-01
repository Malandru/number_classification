import matplotlib.pyplot as plt
import numpy as np


def prepare_figure(numpy_array):
    plt.figure()
    if numpy_array.shape[2] > 1:
        plt.imshow(numpy_array)
    else:
        numpy_array = np.squeeze(numpy_array, axis=-1)
        plt.imshow(numpy_array, cmap='Greys',  interpolation='nearest')


def show_prepared_figures():
    plt.show()
