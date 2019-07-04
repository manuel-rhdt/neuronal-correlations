

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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
    eig_vals_err = np.std(eigen_vals, axis=0)
        
    x = np.arange(1, len(eig_vals_mean) + 1) / (2**index)
    plt.errorbar(x, eig_vals_mean, yerr=eig_vals_err, fmt='o', markersize=1, elinewidth=0.5, label="$K={}$".format(2**index))

plt.yscale('log')
plt.xscale('log')
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




    (array([ 0.84413494, -0.04748481]), array([[2.46931359e-04, 6.13654543e-05],
            [6.13654543e-05, 1.54239868e-05]]))




```python
x = np.linspace(.8, 500, 100)
plt.plot(x, f(x, *popt), '--')
plt.plot(x, f(x, 1.0, popt[1]), '--', linewidth=1, label="independent neurons")
plt.errorbar(cluster_sizes, neg_log(p_zero), fmt='ko')
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
popt
```




    array([ 0.84413494, -0.04748481])




```python
matrix = neurenorm.compute_correlation_coefficients(rdata[5])
matrix = np.abs(matrix - np.identity(len(matrix)))
plt.imshow(matrix, interpolation='nearest')
colorbar = plt.colorbar(ticks = [])
```


![png](Notebook_files/Notebook_14_0.png)

