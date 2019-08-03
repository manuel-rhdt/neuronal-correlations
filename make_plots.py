from matplotlib import pyplot as plt
import numpy as np
import neurenorm
import neurenorm.plots.probability_density
import neurenorm.plots.p_zero
import neurenorm.plots.eigenvalues

strategies = ['correlation', 'random']


def get_data():
    clusterings = {}
    rdata = {}  # < renormalized data
    data = neurenorm.load_data("data.tif")
    for strategy in strategies:
        clusterings[strategy] = neurenorm.cluster_recursive(
            data, 9, clustering_strategy=strategy)
        rdata[strategy] = [neurenorm.apply_clustering(
            data, clusters) for clusters in clusterings[strategy]]

    return data, rdata, clusterings


def make_plots():
    plt.style.use('bmh')
    plt.rcParams["figure.figsize"] = np.array([3.3, 4.5])
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'TeX Gyre Termes'
    plt.rcParams.update({'font.size': 8,
                         'mathtext.fontset': 'stix',
                         'figure.autolayout': True
                         })

    data, rdata, clusterings = get_data()
    fig = neurenorm.plots.probability_density.plot(rdata, strategies)
    fig.savefig('images/probability_density.pdf', dpi=300)
    fig = neurenorm.plots.p_zero.plot(rdata, strategies)
    fig.savefig('images/p_zero.pdf', dpi=300)
    fig = neurenorm.plots.eigenvalues.plot(data, clusterings, strategies)
    fig.savefig('images/eigenvalues.pdf', dpi=300)


if __name__ == "__main__":
    make_plots()
