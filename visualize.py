import matplotlib.pyplot as plt
import numpy as np


def prepare_figure(numpy_array):
    plt.figure()
    h, w, c = numpy_array.shape
    if c == 3:  # RGB numpy array
        plt.imshow(numpy_array)
    else:
        mask = numpy_array[..., 0]
        class_values = numpy_array[..., 1]
        image = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                if mask[(i, j)] > 0:
                    color = get_rgb_color(class_values[(i, j)])
                    image[i, j, 0] = color[0]
                    image[i, j, 1] = color[1]
                    image[i, j, 2] = color[2]
                else:
                    image[i, j, 0] = 255
                    image[i, j, 1] = 255
                    image[i, j, 2] = 255
        plt.imshow(image)


def get_rgb_color(class_value):
    if class_value == 0:
        return np.array([0, 0, 0])  # black
    if class_value == 1:
        return np.array([255, 0, 0])  # red
    if class_value == 2:
        return np.array([0, 255, 0])  # green
    if class_value == 3:
        return np.array([0, 0, 255])  # blue
    if class_value == 4:
        return np.array([255, 255, 0])  # yellow
    if class_value == 5:
        return np.array([255, 102, 0])  # orange
    if class_value == 6:
        return np.array([0, 102, 255])  # purple
    if class_value == 7:
        return np.array([255, 0, 255])  # pink
    if class_value == 8:
        return np.array([102, 204, 255])  # light blue
    if class_value == 9:
        return np.array([102, 51, 0])  # brown


def show_prepared_figures():
    plt.show()
