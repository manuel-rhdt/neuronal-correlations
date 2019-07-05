

```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import json

plt.style.use('bmh')
plt.rcParams["figure.figsize"] = (6/1.5,7/1.5)
plt.rcParams["figure.dpi"] = 200
```


```python
import neurenorm
```


```python
data = neurenorm.load_data("data.tif")
```


```python
# This returns a list of the different renormalization steps
clusterings = neurenorm.cluster_recursive(data, 9, clustering_strategy='random')
rdata = [neurenorm.apply_clustering(data, clusters) for clusters in clusterings]
```


```python
def sorted_eigenvals_for_cluster(data, cluster):
    corr_coef = neurenorm.compute_correlation_coefficients(data[cluster])
    eig_vals, _ = np.linalg.eig(corr_coef)
    return -np.sort(-eig_vals)
```


```python
for index in range(5,8):
    clusters = clusterings[index].T
    eigen_vals = np.zeros_like(clusters, float)
    for jdex, cluster in enumerate(clusters):
        eigen_vals[jdex][...] = sorted_eigenvals_for_cluster(data, cluster)
    
    eig_vals_mean = np.mean(eigen_vals, axis=0)
    eig_vals_err = np.std(eigen_vals, axis=0, ddof=1)
        
    x = np.arange(1, len(eig_vals_mean) + 1) / (2**index)
    plt.errorbar(x, eig_vals_mean, yerr=eig_vals_err, fmt='o', markersize=1, elinewidth=0.5, label="$K={}$".format(2**index))

plt.yscale('log')
plt.xscale('log')
plt.xlim(10**-2, 10**-.5)
plt.xlabel('$\mathrm{rank} / K$')
plt.ylabel('eigenvalue')
plt.title('Scaling of eigenvalues in covariance spectra')
plt.legend(loc=0)
plt.show()
```


![png](Notebook_files/Notebook_5_0.png)



```python
x = np.linspace(0, 1, len(data[0]))

for subdata in rdata[::2]:
    plt.plot(x, subdata[0])
    
plt.xlabel('fractional time')
plt.ylabel('normalized activity')
plt.show()
```


![png](Notebook_files/Notebook_6_0.png)



```python
for i in [0, 4, 6, 7, 8]:
    x, y = neurenorm.make_histogram(rdata[i])
    plt.yscale('log')
    plt.xlim(-0.5,8.5)
    plt.plot(x, y, label="$K={}$".format(2**i))
plt.ylabel('probability density')
plt.xlabel('normalized activity')
plt.legend(loc=0)
plt.ylim(0.6*10**-4, 2)
plt.show()
```


![png](Notebook_files/Notebook_7_0.png)



```python
p_zero, p_errs, cluster_sizes = neurenorm.compute_p_trajectory(rdata[:-1])
```


```python
def neg_log(data):
    return -np.log(data)
```


```python
# Fitting the exponents to the P_0 curve

f = lambda x, beta, a: -a * np.power(x, beta)
popt, perr = curve_fit(f, cluster_sizes, neg_log(p_zero))
```


```python
popt, perr
```




    (array([ 0.87347328, -0.02761152]), array([[3.21897073e-05, 4.66723522e-06],
            [4.66723522e-06, 6.83818876e-07]]))




```python
x = np.linspace(.8, 500, 100)
plt.plot(x, f(x, *popt), 'C0-', label="$-a K^{{{:.2}}}$".format(popt[0]))
plt.plot(x, f(x, 1.0, popt[1]), 'C1:', linewidth=1, label="independent neurons")
plt.errorbar(cluster_sizes, neg_log(p_zero), fmt='C2o')
plt.yscale('log')
plt.xscale('log')
plt.ylim(.02, 13)
plt.xlabel('cluster size $K$')
plt.ylabel('$-\ln(P_0)$')
plt.legend()
plt.show()
```


![png](Notebook_files/Notebook_12_0.png)



```python
matrix = neurenorm.compute_correlation_coefficients(rdata[5])
matrix = np.abs(matrix - np.identity(len(matrix)))
plt.imshow(matrix, interpolation='nearest')
colorbar = plt.colorbar(ticks = [])
```


![png](Notebook_files/Notebook_13_0.png)



```python
regions_filename = './consensus_regions.json' # for other files change path accordingly
with open(regions_filename) as f:
    regions = json.load(f)

dims = [0,0]
for s in regions:
    max_x = np.max(np.array(s['coordinates'])[:,0])
    max_y = np.max(np.array(s['coordinates'])[:,1])
    dims[0] = max_x if max_x > dims[0] else dims[0]
    dims[1] = max_y if max_y > dims[1] else dims[1]
dims = (dims[0] + 1, dims[1] + 1)    

def tomask(coords):
    coords = np.array(coords)
    mask = np.zeros(dims)
    mask[coords.T[0], coords.T[1]] = 1
    return mask

masks = np.array([tomask(s['coordinates']) for s in regions])
```


```python
def cluster_with_color(cluster, masks, color, alpha=None):
    color = matplotlib.colors.to_rgba_array(color, alpha)[0]
    cluster_im = np.sum(masks[cluster], axis=0)
    s = cluster_im.shape
    rgb_im = np.zeros((s[0], s[1], color.shape[0]))
    rgb_im[cluster_im > 0] = color
    return rgb_im
```


```python
# show the outputs
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_facecolor('black')


background = cluster_with_color(np.arange(len(masks)), masks, 'C9', 0.5)
plt.imshow(background)
for cluster_index in range(9):
    color = 'C{}'.format(cluster_index % 9)
    cluster = clusterings[2].T[cluster_index]
    im = cluster_with_color(cluster, masks, color, 0.8)
    plt.imshow(im)
    
plt.show()
```


![png](Notebook_files/Notebook_16_0.png)



```python
masks[0]
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])


