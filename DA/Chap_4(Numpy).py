def main():
    import numpy as np
    # ndarray attributes
    array1 = np.random.randn(2, 3)  # normally distributed random number
    print(array1.shape)
    print(array1.dtype)  # the data shape
    print(array1.ndim)

    # create ndarray
    data = [6, 7.5, 8, 1]
    array = np.array(data, dtype=np.float64)  # transform a sequence into ndarray; syntax like set()
    array = np.zeros(4)
    array = np.zeros((4, 3))  # put a tuple for shape
    array = np.arange(10)  # array version of built-in range()
    array = array.astype(np.string_)  # change the dtype of array; create a new copy of data
    '''array = array.astype(array1)'''  # change the dtype to another ndarray

    # array arithmetic
    array1 = np.array([[1, 2, 3], [4, 5, 6]])
    array2 = np.array([[0, 4, 1], [7, 2, 12]])
    print(array2 > array1)  # computation is carried in-placely

    # index and slice
    name = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    data = np.random.randn(7, 4)
    data[name == 'Bob', 2:]  # it obtain index/slices from the 'name' array; use it ot obtain object from 'data' array
    data[~(name == 'Bob')]  # the '~' symbol negate the argument
    mask = (name == 'Bob') | (name == 'Will')  # use '|' and '&', instead of 'and', 'or'
    data[mask]  # multiple boolean indexing always create new copy of data; unlike what numpy normally do.

    # fancy indexing; index using integer arrays. Copy the arrays
    arr = np.empty((8, 4))
    for i in range(8):
        arr[i] = i
    arr[[4, 3, 0, 6]]  # select particular row with determined order
    arr[[1, 5, 7, 2,], [0, 3, 1, 2]]  # this select only four element at (1, 0), (5, 3), (7, 1), (2, 2)
    arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]  # this select a rectangular area of elements

    # tarnspose and swap
    arr = np.arange(15).reshape((3, 5))
    arr.T  # a view for traditional transform
    np.dot(arr.T, arr)  # compute the inner matrix product
    arr = np.arange(16).reshape((2, 2, 4))
    arr.transpose((1, 0, 2))  # this will permute the axes.
    arr.swapaxes(1, 2)  # this will swap the axes. Change the order of items!

    # unary, binary functions
    arr = np.arange(10)
    np.sqrt(arr)
    np.exp(arr)
    x = np.random.randn(8)
    y = np.random.randn(8)
    np.maximum(x, y)  # output element-wise maximum
    x.argmax()  # fully scan and return the first maximum

    # function on grid
    import matplotlib.pyplot as plt
    points = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(points, points)  # cartesian output
    z = np.sqrt(x ** 2, y ** 2)  # element-wise calculation
    plt.imshow(z, cmap=plt.cm.gray)
    plt.colorbar()
    plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
    plt.show()

    # array methods
    arr = np.random.randn(4, 4)
    np.where(arr > 0, 2, -2)  # a conditional logic to create a new array

    arr = np.random.randn(5, 4)
    arr.mean()
    np.mean(arr)  # the same as the above one
    arr.mean(axis=1)  # compute against particular axis
    arr.sum(axis=0)
    arr = np.arange(0, 9).reshape(3, 3)
    arr.cumsum(axis=0)  # cumulative sum
    arr.cumprod(axis=1)  # cumulartive product

    bools = np.array([False, False, True, False])
    bools.any()
    bools.all()  # also take non-zero value as True

    arr = np.random.randn(6).reshape(3, 2)
    arr.sort(1)  # particular axis sort. It created a sorted copy

    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    np.unique(names)  # the unique value in array. sorted(set(names)
    values = np.array([6, 0, 0, 3, 2, 5 ,6])
    np.in1d(values, [2, 3, 6])  # return boolean of whether values in the latter list

    '''np.save('some_array', arr)
    np.savez('some_array', a=array1, b=array2)  # store multiple arrays in one file
    np.savez_compressed('some_array', a=array1, b=array2)
    arch = np.load('some_array')
    arch['a']  # index the arrays as in a dict.'''

    # Linalg
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[6, 23], [-1, 7], [8, 9]])
    x.dot(y)
    np.dot(x, y)
    x @ y  # these there are the same

    # random
    from random import normalvariate
    samples = [normalvariate(0, 1) for _ in range(16)]  # built-in random
    samples = np.random.normal(loc=0, scale=0.25, size=(4, 4))  # numpy random
    np.random.seed(42)  # a global seed
    rng = np.random.RandomState(42)
    rng.randn(10)  # a private seed with operation under the reference



if __name__ == '__main__':
    main()