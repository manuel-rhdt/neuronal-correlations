""" Make the probability density plot.
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde
from neurenorm import EPSILON


def plot(rdata, strategies):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    for ax, strategy in zip(axs, strategies):
        for i in [0, 4, 5, 6, 7]:
            samples = rdata[strategy][i].flatten()
            samples = samples[samples > EPSILON]
            kde = gaussian_kde(samples, 0.15)
            x = np.linspace(0, 10, 100)
            ax.set_yscale('log')
            ax.set_xlim(-0.5, 8.5)
            ax.plot(x, kde.pdf(x), label="$K={}$".format(2**i), linewidth=1)

        ax.set_xlabel('normalized activity')
        ax.set_ylim(0.6*10**-3, 1.2)
        ax.set_title(strategy + ' clustering')
        ax.set_ylabel('probability density')
        ax.legend(loc=1)

    return fig
