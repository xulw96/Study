def main():
    # read Unicode
    symbols = 'ABCDEF'
    codes = []
    for symbol in symbols:
        codes.append(ord(symbol))
    print("content of codes:", codes)
    # achieve same goal with listcomps
    symbols = 'ABCDEF'
    codes = [ord(symbol) for symbol in symbols]
    print("content of codes:", codes)
    # no variable leakage outside the listcomps
    x = 'ABC'
    dummy = [ord(x) for x in x]
    print("content of x:", x)
    print("content of dummy:", dummy)
    # compare listcomps and map/filter composition
    symbols = 'ABCDEF'
    ascii = [ord(s) for s in symbols if ord(s) < 127]  # no need for 'and' between the two conditions
    print("content of ascii:", ascii)
    ascii = list(filter(lambda c: c < 127, map(ord, symbols)))  # later about this filter/map
    print("content of ascii:", ascii)
    # listcomps for Cartesian product
    colors = ['black', 'white']
    sizes = ['s', 'm', 'l']
    for color in colors:
        for size in sizes:
            print((color, size))
    tshirts = [(color, size) for color in colors for size in sizes]
    print("type of tshirts:", tshirts)
    tshirts = [(color, size) for size in sizes for color in colors]  # change for loop position -> change the group order
    print("type of tshirts(note the group order:", tshirts)
    # genexps (do not generate a whole list to feed others) sequences != lists; item one by one
    symbols = 'ABCED'
    x = tuple(ord(symbol) for symbol in symbols)  # difference lies in parentheses, not bracket
    print("a tuple:", x)
    import array
    x = array.array('I', (ord(symbol) for symbol in symbols))  # first variable grant storage type, the second one uses still paraentheses
    print("an array", x)
    # genexps for Cartesian product
    colors = ['black', 'white']
    sizes = ['s', 'm', 'l']
    for tshirt in ('%s %s' % (c, s) for c in colors for s in sizes):
        print(tshirt)
    # tuples
    traveler_ids = [('USA', '31195855'), ('BRA', 'CE342567'), ('DEU', 'XDA205856')]
    for passport in sorted(traveler_ids):  # iterate over the list and bound passport to tuple.
        print('%s/%s' % passport)  # '%' treat items as fields
    for country, _ in traveler_ids:  # use '_' to ignore the second variable
        print(country)
    # tuple unpacking
    lax_coordinates = (33.9425, -118.408056)
    latitude, longitude = lax_coordinates  # assigning value by unpacking
    print("latitude is:", latitude)
    print("longitude is:", longitude)
    t = (20, 8)
    print("20 divided by 8 is:", divmod(*t))  # calling function during unpacking
    import os
    _, filename = os.path.split('/home/luciano/.ssh/idrsa.pub')
    print("filename is:", filename)
    # *args to grab excess items
    a, b, *rest = range(5)
    print(a, b, rest)
    a, b, *rest = range(2)
    print(a, b, rest)
    a, *body, c, d = range(5)
    print(a, body, c, d)
    *head, b, c, d = range(5)
    print(head, b, c, d)
    # unpack nested tuple
    metro_areas = [
        ('Tokyo', 'JP', 36.933, (35.687922, 139.691667)),  # the last one is a coordinate pair as as one field
        ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
        ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
        ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
        ('Sao Paulo', 'BR', 19.649, (-23.547778, -46.635833))
    ]
    print('{:15} | {:^9} | {:^9}'.format('', 'lat.', 'long.'))
    fmt = '{:15} | {:9.4f} | {:9.4f}'  # setting up format for printing
    for name, cc, pop, (latitude, longitude) in metro_areas:  # assign last value to a tuple
        if longitude <= 0:  # setting condition for output (western hemisphere)
            print(fmt.format(name, latitude, longitude))  # using ".format" to arrange output
    # named tuple
    from collections import namedtuple
    City = namedtuple('City', 'name country population coordinates')
    tokyo = City('Tokyo', 'JP', 36.933, (35.689722, 139.691667))
    print("attribute for tokyo:", tokyo)
    print("tokyo's population:", tokyo[2])
    print("tokyo's population:", tokyo.population)
    # attributes and methods for 'named tuple'
    print("attributes of City class:", City._fields)
    LatLong = namedtuple('LatLong', 'latitude longitude')
    delhi_data = ('Delhi NCR', 'IN', 21.935, LatLong(28.613889, 77.208889))
    delhi = City._make(delhi_data)  # build namedtuple from iterable
    print(delhi)
    delhi = City(*delhi_data)  # another function to do same thing
    print(delhi)
    print("new dict information:", delhi._asdict())  # from namedtuple to dict
    for key, value in delhi._asdict().items():
        print(key + ':', value)
    # slicing
    l = [10, 20, 30, 40, 50, 60]
    print("split at 3:", l[:3], l[3:])
    # assign value to slices
    l = list(range(10))

    # l[2:5] = 100  # won't work. Not iterable object

    l[2:5] = [100]
    print("list is:", l)
    # using * with sequences
    l = [1, 2, 3]
    print("l * 5 = ", l * 5)
    print("5 * 'abcd' = ", 5 * 'abcd')
    # build list
    board = [['_'] * 3 for i in range(3)]  # do with listcomps
    print("board is:", board)
    board[1][2] = ['x']  # for row 1 and column 2
    print("changed board is:", board)

    board = []
    for i in range(3):
        row = ['_'] * 3
        board.append(row)  # the underneath hood for the above. each iteration build a new row

    weird_board = [['_'] * 3] * 3  # note the difference with the last two
    print("weird board is:", weird_board)
    weird_board[1][2] = ['x']
    print("changed weird board is:", weird_board)

    row = ['_'] * 3
    board = []
    for i in range(3):
        board.append(row)  # the underneath hood for the above. Use same row
    # Augmented assignment
    l = [1, 2, 3]  # for a list
    print("id(l) is:", id(l))
    l *= 2
    print("l *= 2 is:", l)
    print("id(l)", id(l))
    t = (1, 2, 3)  # for a tuple
    print("id(t) is", id(t))
    t *= 2
    print("l *= 2 is:", l)
    print("id(t)", id(t))
    # list.sort and sorted
    fruits = ['grape', 'raspberry', 'apple', 'banana']
    print(sorted(fruits))
    print(sorted(fruits, reverse=True))
    print(sorted(fruits, key=len))
    print(sorted(fruits, key=len, reverse=True))  # not a simple reverse, a stable sorting for equal items
    print(fruits)
    print(fruits.sort())  # return None when no new object is created
    print(fruits)  # the original list is changed
    # bisect.bisect
    import bisect
    import sys
    HAYSTACK = [1, 4, 5, 6, 7, 12, 15, 20, 21, 23, 23, 26, 29, 30]
    NEEDLES = [0, 1, 2, 5, 8, 10, 22, 23, 29, 30, 31]
    if sys.argv[-1] == 'left':  # checking the command-line argument
        bisect_fn = bisect.bisect_left  # left for the existing item; right for one after.
    else:
        bisect_fn = bisect.bisect
    print('DEMO:', bisect_fn.__name__)  # print name of the function
    print('haystack ->', ' '.join('%2d' % n for n in HAYSTACK))
    ROW_FMT = '{0:2d} @@ {1:2d}  {2}{0:2d}'  # set format for output
    for needle in reversed(NEEDLES):
        position = bisect_fn(HAYSTACK, needle)  # get the insertion position
        offset = position * ' | '  # vertical bars at insertion point
        print(ROW_FMT.format(needle, position, offset))
    # perform table lookup
    def grade(score, breakpoints = [60, 70, 80, 90], grades='FDCBA'):
        i = bisect.bisect(breakpoints, score)  # default for bisect is 'right'
        return grades[i]
    print([grade(score) for score in [33, 99, 77, 70, 89, 90, 100]])
    # bisect.insort
    import bisect
    import random
    SIZE = 7
    random.seed(42)
    my_list = []
    for i in range(SIZE):
        new_item = random.randrange(SIZE*2)
        bisect.insort(my_list, new_item)  # insort will keep sequnce being sorted
        print('%2d ->' % new_item, my_list)
    # interact with arrays
    """from array import array
    from random import random
    floats = array('d', (random() for i in range(10**7)))  # double-precision floats
    print("last item for floats:", floats[-1])
    fp = open('floats.bin', 'wb')
    floats.tofile(fp)  # save the file
    fp.close()
    floats2 = array('d')  # empty array of doubles
    fp = open('floats.bin', 'rb')
    floats2.fromfile(fp, 10**7)  # read the file
    fp.close()
    print("last item for floats2:", floats2[-1])"""
    # memoryview
    numbers = array.array('h', [-2, -1, 0, 1, 2])  # short signed integers
    memv = memoryview(numbers)
    print("first character for the memory:", memv[0])
    memv_oct = memv.cast('B')  # unsigned char
    print("inspect memv_oct", memv_oct.tolist())
    memv_oct[5] = 4  # assign value 4
    print("changed list:", numbers)
    # basic numpy
    import numpy
    a = numpy.arange(12)
    print(a)
    print(type(a))
    print(a.shape)
    a.shape = (3, 4)
    print(a.shape)
    print(a[2])
    print(a[2, 1])
    print(a[:, 1])
    print(a.transpose())
    # high-level numpy
    import numpy
    """floats = numpy.loadtxt('floats-10M-lines.txt')
    print(floats[-3:])
    floats *= 5
    print(floats[-3:])""" # numpy.array can hold non-number values
    # Deques
    from collections import deque
    dq = deque(range(10), maxlen=10)
    print(dq)
    dq.rotate(3)
    print(dq)
    dq.extend([11, 22, 33])  # discard items when above the maxlen
    print(dq)
    dq.appendleft([10, 20, 30, 40])  # adding list as if one element
    print(dq)
    dq.extendleft([10, 20, 30, 40])  # adding by iterating; so the order is reversed
    print(dq)



if __name__ == "__main__":
    main()
