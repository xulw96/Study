def main():
    # coroutines
    def averager():
        total = 0.0
        count = 0
        average = None
        while True:
            term = yield average  # everytime output an average and get a new term from client
            total += term
            count += 1
            average = total/count
    # decorator to prime the coroutine
    from functools import wraps
    def coroutines(func):
        """Decorator: primes 'func' by advancing to first 'yield' """
        @wraps(func)
        def primer(*args, **kwargs):
            gen = func(*args, **kwargs)
            next(gen)
            return gen
        return primer
    # exception handling
    class DemoException(Exception):
        """An exception type for demonstration"""
    def demo_exc_handling():
        print('-> coroutine started')
        while True:
            try:
                x = yield
            except DemoException:
                print('*** DemoException handled. Continuing...')
            else:
                print('-> coroutine received: {!r}'.format(x))
        raise RuntimeError('This line should never run')
    # add action upon coroutines termination
    def demo_finally():
        print(' -> coroutine started')
        try:  # a try/finally closure to handle termination
            while True:
                try:
                    x = yield
                except DemoException:
                    print('*** DemoException handled. Continuing...')
                else:
                    print('-> coroutine received: {!r}'.format(x))
        finally:
            print('-> coroutine ending')
    # return a value from coroutine
    from collections import namedtuple
    Result = namedtuple('Result', 'count average')  # to consume the returned value with a namedtuple
    def averager():
        total = 0.0
        count = 0
        average = None
        while True:
            term = yield  # it doesn't yield value by each iteration
            if term is None:  # to return a value, coroutine has to end normally, hence this if-loop
                breal
            total += term
            count += 1
            average = total/count
        return Result(count, average)
    # 'yield from'
    Result = namedtuple('Result', 'count average')
    data = {
        'girls;kg':
            [40.9, 38.5, 44.3, 42.2, 45.2, 41.7, 44.5, 38.0, 40.6, 44.5],
        'girls;m':
            [1.6, 1.51, 1.4, 1.3, 1.41, 1.39, 1.33, 1.46, 1.45, 1.43],
        'boys;kg':
            [39.0, 40.8, 43.2, 40.8, 43.1, 38.6, 41.4, 40.6, 36.3],
        'boys;m':
            [1.38, 1.5, 1.32, 1.25, 1.37, 1.48, 1.25, 1.49, 1.46],
    }
    def report(results):
        for key, result in sorted(results.items()):
            group, unit = key.split(';')
            print('{:2}{:5} averaging {:.2f}{}'.format(result.count, group, result.average, unit))
    def averager():
        total = 0.0
        count = 0
        average = None
        while True:
            term = yield
            if term is None:
                break
            total += term
            count += 1
            average = total/count
        return Result(count, average)  # this value will be passed to client, not delegating generator
    def grouper(results, key):
        while True:
            results[key] = yield from averager()
    def get_result(data):
        results = {}
        for key, values in data.items():
            group = grouper(results, key)
            next(group)
            for value in values:
                group.send(value)  # this is sent directly into averager's term = yield line
            group.send(None)  # terminate the current averager()
        report(results)
    get_result(data)  # call the above functions to run
    # taxi process simulator
    import queue
    def taxi_process(ident, trips, start_time=0):
        """Yield to simulator issuing event at each stage change"""
        time = yield Event(start_time, ident, 'leave garage')  # wait for the compute_delay process to send current time
        for i in range(trips):
            time = yield Event(time, ident, 'pick up passenger')
            time = yield Event(time, ident, 'drop off passenger')
        yield Event(time, ident, 'going home')  # when it ends, 'StopIteration' will be raised
    class Simulator:
        def __init__(self, proc_map):
            self.events = queue.PriorityQueue()
            self.procs = dict(proc_map)
        def run(self, end_time):
            """Schedule and display events until time is up"""
            for _, proc in sorted(self.procs.items()):  # the key is not a matter, so is assigne to '_'
                first_event = next(proc)
                self.events.put(first_event)
            sim_time = 0
            while sim_time < end_time:
                if self.events.empty():  # if the events is empty, the loop ends
                    print('*** end of events ***')
                    break
                current_event = self.events.get()  # return the smallest time in the queue
                sim_time, proc_id, previous_action = current_event  # update simulation clock by unpacking
                print('taxt:', proc_id, proc_id * ' ', current_event)
                active_proc = self.procs[proc_id]
                next_time = sim_time +compute_duration(previous_action)
                try:
                    next_event = active_proc.send(next_time)  # send time to/drive the taxi coroutine
                except StopIteration:
                    del self.proc[proc_id]
                else:
                    self.events.put(next_event)
            else:
                msg = '*** end of simulation time: {} events pending ***'
                print(msg.format(self.events.qsize()))


if __name__ == "__main__":
    main()