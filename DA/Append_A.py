import numpy as np
import pandas as pd

# array manipulation
arr = np.arange(3)
arr.repeat(3)
np.tile(arr, (2, 1))

# braodcasting
arr = np.random.randn(4, 3)
row_means = arr.mean(1)
row_means.shape
shaped = row_means.reshape((4, 1))
arr - shaped  # must reshape to (4, 1) before broadcasting; (4, ) doesn't work

arr = np.zeros((4, 4))
arr_3d = arr[:, np.newaxis, :]  # np.newaxis to build a higher dimension
arr_1d = np.random.randn(size=3)
arr_1d[:, np.newaxis]
arr_1d[np.newaxis, :]

arr = np.zeros((4, 3))
col = np.array([1.28, -0.42, 0.44, 1.6])
arr[:2] = col[:, np.newaxis]  # set value by broadcasting; ensure valid shape

# advanced ufunc
arr = np.arange(10)
np.add.reduce(arr)  # one value output
np.add.accumulate(arr)  # preserving all values along adding
np.logical_and.reduce(arr[1: -1] < arr[: -1], axis=1)  # same to all method
x, y = np.random.randn(3, 4), np.random.randn(5)
np.substract.outer(x, y).shape()  # outer have sum of dimensions
arr = np.arange(10)
np.add.reduceat(arr, [0, 5, 8])  # specify indices for traversing

def add_element(x, y):
    return x + y
add_them = np.frompyfunc(add_element, 2, 1)  # build a ufunc
add_them = np.vectorize(add_element, otypes=[np.float64])  # specify dtype

# structured and record array
dtype = [('x', np.float64), ('y', np.int32)] # name and type
saar = np.array([(1.5, 6), (3.14, -2)], dtype=dtype)  # heterogeneous
sarr['x']  # access by dtype.names
dtype = [('x', np.int64, 3), ('y', np.int32)]
arr =np.zeros(4, dtype=dtype)  # x for length-3 array

# sorting
arr = np.random.randn(6)
np.sort(arr)  # a new sorted copy
arr.sort()  # in-place sort
arr[:, ::-1]  # trick for descending sort

values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()  # return indices from original on a sorted view
values[indexer]  # get the sorted array
arr = np.random.randn(3, 5)
arr[0] = values
arr[:, arr[0].argsort()]

arr = np.random.randn(20)
np.partition(arr, 3)  # the first three is the smallest.

data = np.floor(np.random.uniform(0, 10000, size=50))  # floor() round to integer
bins = np.array([0, 100, 1000, 5000, 10000])
labels = bins.searchsorted(data)  # get indices to insert and keeping array sorted
pd.Series(data).groupby(labels).mean()  # using the indices for grouping into bins

# Numba
import numba as nb
@nb.jit  # used to improve performance
def mean_distance(x, y):
    nx = len(x)
    result = 0
    count = 0
    for i in range(nx):
        result += x[i] - y[i]
        count += 1
    return result / count

@nb.vectorize
def nb_add(x, y):
    return x + y
nb_add.accumulate(x, 0)  # works like built-in ufuncs

# advanced IO
mmap = np.memmap('mymmap', dtype='float64', mode='w+', shape=(100000, 10000))
mmap[:5] = np.random.randn(5, 10000)  # buffered in memory
mmap.flush()  # push to disk
del mmap

arr_f = np.ones((1000, 1000), order='F')
aff_f.copy('C').flags  # create new copy in row-major order
