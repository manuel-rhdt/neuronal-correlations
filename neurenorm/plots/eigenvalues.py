
import neurenorm
import numpy as np
from matplotlib import pyplot as plt


def sorted_eigenvals_for_cluster(data, cluster):
    corr_coef = neurenorm.compute_correlation_coefficients(data[cluster])
    eig_vals, _ = np.linalg.eig(corr_coef)
    return -np.sort(-eig_vals)


def max_likelihood_exp(x):
    xmin = np.min(x)
    mu = 1 + np.size(x) * np.reciprocal(np.sum(np.log(x/xmin)))
    return mu

def plot(data, clusterings, strategies):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    power_law_max = 0.7
    power_law_offset = [0.7, 0.9]

    for plot_i, (ax, strategy) in enumerate(zip(axs, strategies)):
        mu_arr = []
        for index in range(4, 8):
            size_k = 2**index
            clusters = clusterings[strategy][index].T
            eigen_vals = np.zeros_like(clusters, float)
            for jdex, cluster in enumerate(clusters):
                eigen_vals[jdex][...] = sorted_eigenvals_for_cluster(
                    data, cluster)

            # eigenvalues and errors
            eig_vals_mean = np.mean(eigen_vals, axis=0)
            eig_vals_err = np.std(eigen_vals, axis=0, ddof=1)

            # parameter estimation
            rank = np.arange(1, len(eig_vals_mean) + 1)
            power_law = eig_vals_mean[(rank / size_k) < power_law_max]
            mu_arr.append(1.0/max_likelihood_exp(power_law))

            # plotting
            ax.errorbar(rank / size_k, eig_vals_mean, yerr=eig_vals_err, fmt='o',
                        markersize=1, elinewidth=0.5, label="$K={}$".format(size_k))

        mu = np.mean(mu_arr, axis=0)
        x = np.linspace(1/2**7, power_law_max, 10)
        ax.plot(x, power_law_offset[plot_i] * np.power(1.0/x, mu),
                label="$\\mu = {{{:.2}}}$".format(mu), linewidth=1)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(0.3, 7)
        ax.set_xlabel('$\mathrm{rank} / K$')
        ax.set_ylabel('eigenvalue')
        
        ax.set_title(strategy + ' clustering')
        ax.legend(loc=0)

    return fig
