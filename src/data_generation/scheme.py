import random

import numpy as np
import matplotlib.pyplot as plt


def gen_gauss_line(count, x_mean, x_std, series_time=None):
    if series_time is None:
        return [random.gauss(x_mean, x_std) for _ in range(count)]
    return [[random.gauss(x_mean, x_std) for _ in range(series_time)] for _ in range(count)]


def gen_gauss_scheme(scheme, lines_x_mean, lines_x_std, series_time=None, flatten=True):
    positions = [gen_gauss_line(count, x_mean, x_std, series_time)
                 for count, x_mean, x_std in zip(scheme, lines_x_mean, lines_x_std)]

    if flatten:
        positions = [item for sublist in positions for item in sublist]

    return positions


def gen_scheme(scheme, x_min=0, x_max=100, std_divider=8, series_time=None, flatten=True):
    lines_num = len(scheme)
    line_width = (x_max - x_min) // lines_num

    lines_x_mean = [(2 * i + 1) * line_width / 2 for i in range(lines_num)]
    lines_x_std = [line_width / std_divider for i in range(lines_num)]

    return gen_gauss_scheme(scheme, lines_x_mean, lines_x_std, series_time, flatten)


def plot_scheme(positions):
    if isinstance(positions[0], list):
        positions = [item for sublist in positions for item in sublist]

    for p in positions:
        plt.scatter(p, 0)
    plt.show()


def plot_series(positions):
    for p in positions:
        plt.scatter(p, 0, c='b')
    plt.scatter(np.mean(positions), 0, c='r')
    plt.show()


if __name__ == '__main__':
    # line = gen_gauss_line(4, 10, 3)
    # print(line)

    # scheme = gen_gauss_scheme([4, 4, 2], [0, 20, 40], [3, 3, 3], False)
    # print(scheme)
    # plot_scheme(scheme)

    for _ in range(10):
        s = gen_scheme([4, 3, 3], flatten=False)
        print(s)
        plot_scheme(s)
