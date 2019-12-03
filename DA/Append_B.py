def main():
    from IPython.core.debugger import Pdb
    def set_trace():
        Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)
    def debug(f, *args, **kwargs):
        pdb = Pdb(color_scheme='Linux')
        return pdb.runcall(f, *args, **kwargs)

    import time
    start = time.time()
    iterations = 5
    for i in range(iterations):
        pass
    elapsed_per = (time.time() - start) / iterations

    strings = ['foo', 'foobar', 'baz', 'qux',
               'python', 'Guido Van Rossum'] * 100000
    method1 = [x for x in strings if x.startswith('foo')]
    method2 = [x for x in strings if x[:3] == 'foo']

    import numpy as np
    from numpy.linalg import eigvals
    def run_experiment(niter=100):
        k = 100
        results = []
        for _ in range(niter):
            mat = np.random.randn(k, k)
            max_eigenvalue = np.abs(eigvals(mat)).max()
            results.append(max_eigenvalue)
        return results
    some_results = run_experiment()
    print('Largest one we saw: %s' % np.max(some_results))

    def add_and_sum(x, y):
        added = x + y
        summed = added.sum(axis=1)
        return summed
    def call_function():
        x = np.random.randn(1000, 1000)
        y = np.random.randn(1000, 1000)
        return add_and_sum(x, y)



if __name__ == '__main__':
    main()
