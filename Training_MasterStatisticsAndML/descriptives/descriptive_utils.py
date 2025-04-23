"""
Central Tendency Representations:
    - mean - bias because of outliers
    - median - 50% of values as lower or equal
    - mode - most frequent appears value/values
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import random


def generate_gauss_dist(n, shift=0., width=1.):
    """
    :param n: No. samples
    :param shift: correspond to mean
    :param width: correspond to std
    :return: Gauss distribution
    """
    return width*np.random.randn(n) + shift


def generate_log_norm_dist(n, shift=0., width=1.):
    return np.exp(width*np.random.randn(n) + shift)


def calc_hist(dist, nbins):
    """
    :param dist: distribution/data to calc hist
    :param nbins: bins
    :return: histogram of data
    """
    y, x = np.histogram(dist, nbins)
    x = (x[:-1] + x[1:]) / 2  # calc the x-center of each bin
    return x, y


def display_hist_line_plot(subdists: tuple, nbins):
    """
    :param subdists: distribution/data to display
    :param nbins: bins
    :return: display histograms
    """
    for i, dist in enumerate(subdists):
        x, y = calc_hist(dist, nbins)
        mean = np.round(np.mean(dist), 4)
        c = (random.random(), random.random(), random.random())
        plt.plot(x, y, color=c, label=f"gauss_mean={mean}")
        plt.plot([mean, mean], [0, max(y)], '--', color=c)

    plt.xlabel('Data Values')
    plt.ylabel('Data Counts')
    plt.legend()
    plt.show()


def display_central_tendency(dist, nbins, outliers=False):
    x, y = calc_hist(dist, nbins)
    if outliers:
        dist = np.append(dist, [10000, 20000])
        x = np.append(x, [10000, 20000])
        y = np.append(y, [1, 1])
    mean = np.round(np.mean(dist), 4)
    median = np.round(np.median(dist), 4)
    mode = x[np.argmax(y)]

    plt.plot(x, y)
    plt.plot([mean, mean], [0, max(y)], '--b', label=f"dist_mean={mean}")
    plt.plot([median, median], [0, max(y)], '--r', label=f"dist_median={median}")
    plt.plot([mode, mode], [0, max(y)], '--k', label=f"dist_mode={mode}")

    plt.xlabel('Data Values')
    plt.ylabel('Data Counts')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    nsamp = 10001
    n_bins = 30

    # compare distributions
    d1 = generate_gauss_dist(nsamp, shift=-1)
    d2 = generate_gauss_dist(nsamp, width=3)
    d3 = generate_gauss_dist(nsamp, shift=1)
    display_hist_line_plot((d1, d2, d3), n_bins)

    # combine gauss distributions
    d4 = np.hstack((generate_gauss_dist(nsamp, shift=-2),
                    generate_gauss_dist(nsamp, shift=2)))
    display_hist_line_plot((d4,), n_bins)
    display_central_tendency(d4, n_bins)

    # normal-log distributions
    d5 = generate_log_norm_dist(nsamp, width=0.7)
    display_hist_line_plot((d5,), nbins=50)
    display_central_tendency(d5, nbins=50)

    # compare distributions
    d6 = generate_gauss_dist(10000, shift=0, width=1)
    display_central_tendency(d6, nbins=50)
    display_central_tendency(d6, nbins=50, outliers=True)

    d7 = generate_gauss_dist(20, shift=0, width=1)
    display_central_tendency(d7, nbins=20)
    display_central_tendency(d7, nbins=20, outliers=True)
