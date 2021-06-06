import matplotlib.pyplot as plt
import numpy as np


# https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
def n_bins(column):
    q25, q75 = np.percentile(column, [.25, .75])
    bin_width = 2 * (q75 - q25) * len(column) ** (-1 / 3)
    bins = round((column.max() - column.min()) / bin_width)
    return bins


# https://stackoverflow.com/questions/33203645/how-to-plot-a-histogram-using-matplotlib-in-python-with-a-list-of-data
def plot_auto(column):
    b = n_bins(column)
    plt.hist(column, density=True, bins=b)
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.show()
