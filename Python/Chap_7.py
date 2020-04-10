def main():
    # decorator 101
    def deco(func):
        def inner():
            print('running inner()')
        return inner  # decorator will retrun a function back
    @deco
    def target():
        print('running target()')
    print(target())  # function being replaced and actually runs inner
    print(target)  # works as a reference to 'inner'

    # python execute decorator
    registry = []  # hold rference to functions being decorated
    def register(func):  # function as argument
        print('running register(%s)' % func)  # decorator runs/imports before any other function!
        registry.append(func)
        return func
    @register
    def f1():
        print('running f1()')
    @register
    def f2():
        print('running f2()')
    def f3():
        print('running f3()')
    print('running main()')
    print('registry ->', registry)
    f1()
    f2()
    f3()

    # best_promo list with decorator
    promos = []
    def promotion(promo_func):
        promos.append(promo_func)
        return promo_func
    @promotion  # the decorator highlight purpose of the function
    def fidelity(order):
        """5% discount for customers with 1000 or more fidelity points"""
        return order.total() * 0.05 if order.customer.fidelity >= 1000 else 0
    @promotion
    def bulk_item(order):
        """10% discount for each LineItem with 20 or more units"""
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * 0.1
        return discount
    @promotion
    def large_order(order):
        """7% discount for orders with 10 or more distinct items"""
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * 0.07
        return 0
    def best_promo(order):
        """select the best disocunt available"""
        return max(promo(order) for promo in promos)

    # variable scope
    b = 9
    def f1(a):
        print(a)
        print(b)
        b = 3  # this assignment makes 'b' a local variable
        '''f1(3)'''  # won't work
    def f2(a):
        global b
        print(a)
        b = 9
        print(b)
    f2(3)

    # a running average by class
    class Averager():
        def __init__(self):
            self.series = []
        def __call__(self, new_value):
            self.series.append(new_value)
            total = sum(self.series)
            return total/len(self.series)
    avg = Averager()  # an instance of the class
    print(avg(10), avg(11), avg(12))  # computing average value at running time
    # a running average by higher-order function
    def make_averager():
        series = []  # a free variable
        def averager(new_value):
            series.append(new_value)
            total = sum(series)
            return total/len(series)
        return averager
    avg = make_averager()  # an inner function
    print(avg(10), avg(11), avg(12))
    print(avg.__code__.co_varnames, avg.__code__.co_freevars, sep='\n')
    print(avg.__closure__, avg.__closure__[0].cell_contents, sep='\n')  # the binding for series is kept in the returned function and can be found in cell_contents

    # nonlocal declaration
    def make_averager():
        count = 0
        total = 0
        def averager(new_value):
            nonlocal count, total  # numbers are immutable so the '+=' operator actually implements a local variable
            count += 1
            total += new_value
            return total/count
        return averager

    # clock decorator
    import time
    def clock(func):
        def clocked(*args):  # accept any positional argument
            t0 = time.perf_counter()
            result = func(*args)  # 'func' as a free varialbe
            elapsed = time.perf_counter() - t0
            name = func.__name__
            arg_str = ', '.join(repr(arg) for arg in args)
            print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
            return result
        return clocked  # replace the decorated function
    @clock
    def snooze(seconds):
        time.sleep(seconds)  # delay execution for a given time
    @clock
    def factorial(n):
        return 1 if n < 2 else n*factorial(n-1)
    print('*' * 40, 'Calling snooze(.123)')
    snooze(0.123)
    print('*' * 40, 'Calling factorial(6)')
    print('6!=', factorial(6))

    # functools.wraps
    import functools
    def clock(func):
        @functools.wraps(func)  # decorator provided by functools module
        def clocked(*args, **kwargs):  # support keywords arguments
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            name = func.__name__
            arg_lst = []
            if args:
                arg_lst.append(', '.join(repr(arg) for arg in args))
            if kwargs:
                pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.itmes())]
                arg_lst.append(', '.join(pairs))
            arg_str = ', '.join(arg_lst)
            print('[%0.8fs] %s(%s) -> %r ' % (elapsed, name, arg_str, result))
            return result
        return clocked

    # lru_cache
    import functools
    @functools.lru_cache()  # have parenthese because it contains arguments (maxsize=128, typed=False)
    @clock  # stacked decorator
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n-2) + fibonacci(n-1)  # avoid waste in recursive calling
    print(fibonacci(6))

    # singledispatch and generic function
    from functools import singledispatch
    from collections import abc
    import numbers
    import html
    @singledispatch  # bound several functions into one
    def htmlize(obj):
        content = html.escape(repr(obj))
        return '<pre>{}</pre>'.format(content)
    @htmlize.register(str)  # use register to differentiate function calls with regard to types (generated by 'singledispatch')
    def _(text):  # the name for specialized function is irrelevant
        content = html.escape(text).replace('\n', '<br>\n')
        return '<p>{0}</p>'.format(content)
    @htmlize.register(numbers.Integral)
    def _(n):
        return '<pre>{0} (0x{0:x})</pre>'.format(n)
    @htmlize.register(tuple)  # stack decorator to support different types
    @htmlize.register(abc.MutableSequence)
    def _(seq):
        inner = '</li>\n<li>'.join(htmlize(item) for item in seq)
        return '<ul>\n<li>' + inner + '</li>\n</ul>'
    print(htmlize({1, 2, 3}))
    print(htmlize(abs))
    print(htmlize('Heimlich & Co.\n- a game'))
    print(htmlize(42))
    print(htmlize(['alpha', 66, {3, 2, 1}]))

    # parameterized decorators
    registry = set()  # 'set' is faster to add and remove
    def register(active=True):
        def decorate(func):  # decorator becomes function to accept parameters
            print('running register(active=%s) -> decorate(%s)' % (active, func))
            if active:
                registry.add(func)
            else:
                registry.discard(func)
            return func  # the decorator return the function
        return decorate  # the decorator factory return the decorator
    @register(active=False)
    def f1():
        print('running f1()')
    @register()  # be called as a funtion to return the decorator
    def f2():
        print('running f2()')
    def f3():
        print('running f3()')
    f1()
    f2()
    f3()
    print(registry)
    register()(f3)  # returns decorate, which is then applied to f3
    print('add f3', registry)
    register(active=False)(f2)
    print('remove f2', registry)

    # parameterized clock decorator
    import time
    DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'
    def clock(fmt=DEFAULT_FMT):
        def decorate(func):
            def clocked(*_args):
                t0 = time.time()
                _result = func(*_args)
                elapsed = time.time() - t0
                name = func.__name__
                args = ', '.join(repr(arg) for arg in _args)
                result = repr(_result)
                print(fmt.format(**locals()))  # '**locals()' allow local variable to be referenced
                return _result
            return clocked
        return decorate
    @clock()
    def snooze(seconds):
        time.sleep(seconds)
    for i in range(3):
        snooze(0.123)
    @clock('{name}: {elapsed}s')  # input argument to change the format
    def snooze(seconds):
        time.sleep(seconds)
    for i in range(3):
        snooze(0.123)
    @clock('{name}({args}) dt={elapsed:0.3f}s')   # input argument to change the format
    def snooze(seconds):
        time.sleep(seconds)
    for i in range(3):
        snooze(0.123)


if __name__ == "__main__":
    main()