""" Produce the P_0 plot
"""

import neurenorm
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def neg_log(data):
    return -np.log(data)


def plot(rdata, strategies):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    for plot_i, (ax, strategy) in enumerate(zip(axs, strategies)):
        p_zero, p_errs, cluster_sizes = neurenorm.compute_p_trajectory(
            rdata[strategy][:-1])

        # Fitting the exponents to the P_0 curve

        def f(x, beta, a): return -a * np.power(x, beta)
        popt, perr = curve_fit(f, cluster_sizes, neg_log(p_zero))

        x = np.linspace(.8, 500, 100)
        if plot_i == 0:
            ax.plot(x, f(x, *popt), 'C0-',
                    label="$-a K^{{{:.2}}}$".format(popt[0]))
        ax.plot(x, f(x, 1.0, -0.02761152), 'C1:',
                linewidth=1, label="indep\nneurons")
        ax.errorbar(cluster_sizes, neg_log(p_zero), fmt='C2o')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(.02, 13)
        ax.set_xlabel('cluster size $K$')
        ax.legend(loc=2)
        ax.set_title(strategy + ' clustering')
        ax.set_ylabel('$-\ln(P_0)$')

    return fig
