import numpy as np


def main():
    def append_element(some_list, element):
        some_list.append(element)
    data = [1, 2, 3]  # list has an attribute of append
    append_element(data, 4)
    print("data is:", data)
    # check type
    a = 5
    print(isinstance(a, int))
    print(type(None))
    # duck typing´®
    def isiterable(obj):
        try:
            iter(obj)
            return True
        except TypeError:  # not iterable
            return False
    print(isiterable(5))  # check whether object can be iterated
    x = 3
    if not isinstance(x, list) and isiterable(x):
        x = list(x)  # convert x to a list when necessary
    # 'is' and '=='
    a = [1, 2, 3]
    b = list(a)
    print(a is b)  # not the same object
    print(a == b)  # the same value
    # strings
    s = '12\\34'
    print(s)
    s = '12\34'  # '\' works to escape
    print(s)
    s = r'this\has\no\special\characters'  # 'r' for raw
    print(s)
    template = '{0:.2f} {1:s} are worth {2:d}'  # '2f' for floats with two decimal; 's' for string; 'd' for integer
    print(template.format(4.5560, 'argentine pesos', 1))
    # dates and times
    from datetime import datetime, date, time
    dt = datetime(2011, 10, 29, 20, 30, 21)
    print(dt.day)
    print(dt.minute)
    print(dt.date())
    print(dt.strftime('%m/%d/%Y %H:%M'))  # setting up output format with strftime
    date = datetime.strptime('20091031', '%Y%m%d')  # transform output format with strptime(parse)
    print(date)
    dt = dt.replace(minute=0, second=0)
    print("dt is:", dt)
    dt2 = datetime(2011, 11, 15, 22, 30)
    delta = dt2 - dt  # operate on datetime
    print(delta)
    print(type(dt))
    print(type(delta))
    dt = dt + delta  # datetime can add with timedelta, not datetime .
    print(dt)
    # if, pass, elif and else
    if x < 0:
        print("It's negative")
    elif x == 0:
        pass
    elif 0 < x < 5:
        print("Positive but smaller than 5")
    else:
        print("Positive and larger than or equal to 5")
    # for loop, continue and break
    sequence = [1, 2, None, 4, None, 5]
    total = 0
    for value in sequence:
        if value is None:
            continue
        total =+ value
    sequence = [1, 2, 0, 4, 6, 5, 2, 1]
    total_until_5 = 0
    for value in sequence:
        if value == 5:
            break
        total_until_5 += value
    for i in range(4):
        for j  in range(4):
            if j > i:
                break
            print((i, j))
    # tenary expression
    x = 5
    P = "non negative" if x >= 0 else "negative"
    print(P)

if __name__ == "__main__":
    main()
