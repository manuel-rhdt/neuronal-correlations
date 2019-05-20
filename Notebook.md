

```python
import matplotlib.pyplot as plt
import numpy as np
```


```python
import neurenorm
```


```python
# this cell is used to quickly reload the python file
from importlib import reload
reload(neurenorm)
```




    <module 'neurenorm' from '/Users/mr/Uni/neuronal-correlations/neurenorm.py'>




```python
data = neurenorm.load_data("data.tif")
```


```python
rdata = neurenorm.perform_renormalization(data, times = 8)
```


```python
x = np.linspace(0, 1, len(data[0]))

for subdata in rdata[::2]:
    plt.plot(x, subdata[0])
    
plt.show()
```


![png](Notebook_files/Notebook_5_0.png)



```python
for subdata in rdata[::2]:
    x, y = neurenorm.make_histogram(subdata)
    plt.yscale('log')
    plt.xlim(-0.5,8.5)
    plt.plot(x, y)
plt.show()
```


![png](Notebook_files/Notebook_6_0.png)



```python
p_zero, p_errs, cluster_sizes = neurenorm.renormalize_and_compute_p(data)
```


```python
def neg_log(data):
    return -np.log(data)
```


```python
errs = neg_log(p_zero + p_errs / 2) - neg_log(p_zero - p_errs / 2)
plt.errorbar(cluster_sizes, neg_log(p_zero), yerr=errs, fmt='o')
plt.yscale('log')
plt.xscale('log')
```


![png](Notebook_files/Notebook_9_0.png)

