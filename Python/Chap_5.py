def main():
    # function as object
    def factorial(n):
        '''return n!'''
        return 1 if n < 2 else n * factorial(n-1)
    print(factorial(42))
    print(factorial.__doc__)  # '__doc__' attribute return the help text
    print(type(factorial))  # the class 'function'
    fact = factorial  # function as a variable
    print(fact)
    print(map(factorial, range(11)))  # 'map' create a list of results with input function and variables
    print(list(map(fact, range(11))))
    # higher-order functions
    fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana',]
    print(sorted(fruits, key=len))  # one-argument function as key
    def reverse(word):
        return word[::-1]
    print(sorted(fruits, key=reverse))  # passing runtime-made function as key
    # __call__
    import random
    class BingoCage:
        def __init__(self, items):
            self._items = list(items)  # build a local copy
            random.shuffle(self._items)
        def pick(self):
            try:
                return self._items.pop()
            except IndexError:
                raise LookupError('pick from empty BingoCage')
        def __call__(self):
            return self.pick()
    bingo = BingoCage(range(3))
    print(bingo.pick())
    print(bingo())
    print(callable(bingo))
    # function inspection
    print(dir(factorial))  # see attributes for a function
    class c: pass
    def func(): pass  # bare class and function
    obj = c()
    a = sorted(set(dir(func)) - set(dir(obj)))
    print(a)
    # function parameters
    def tag(name, *content, cls=None, **attrs):  # arguments after * will all be keyword argument
        '''Generate one or more HTML tags'''
        if cls is not None:  # a workaround to grant class attribute
            attrs['class'] = cls
        if attrs:
            attr_str = ''.join(' %s="%s"' % (attr, value) for attr, value in sorted(attrs.items()))
        else:
            attr_str = ''
        if content:
            return '\n'.join('<%s%s>%s</%s>' %
                             (name, attr_str, c, name) for c in content)
        else:
            return '<%s%s />' % (name, attr_str)
    print("tag('br')", tag('br'))
    print("tag('p', 'hello'", tag('p', 'hello'))
    print(tag('p', 'hello', 'world'))
    print(tag('p', 'hello', id=33))
    print(tag('p', 'hello', 'world', cls='sidebar'))
    print(tag(content='testing', name='img'))  # the positional 'name' being passed as keyword
    my_tag = {'name': 'img', 'title': 'Sunset Boulevard', 'src': 'sunset.jpg', 'cls': 'framed'}
    print(tag(**my_tag))  # prefix '**' pass items as separate arguments to corresponding parameters
    # info about parameters
    def clip(text, max_len=80):
        '''return text clipped at the last space before or after max_len'''
        end = None
        if len(text) > max_len:
            space_before = text.rfind(' ', 0, max_len)
            if space_before >= 0:
                end =space_before
            else:
                space_after = text.rfind(' ', max_len)
                if space_after >= 0:
                    end = space_after
        if end is None:  # no spaces were found
            end = len(text)
        return text[:end].rstrip()
    print(clip.__defaults__)  # default value need to be paired to arguments
    print(clip.__code__)
    print(clip.__code__.co_varnames)
    print(clip.__code__.co_argcount)
    # function signature
    from inspect import signature
    sig = signature(clip)  # create an inspect.Signature object.
    print(sig)
    print(str(sig))
    for name, param in sig.parameters.items():  # Each parameter has 'name', 'default' and 'kind'
        print(param.kind, ':', name, '=', param.default)  # no default parameter has class of  'inspect._empty'
    # bind arguments into parameters
    sig = signature(tag)
    bound_args = sig.bind(**my_tag)  # passing a dict of arguments to '.bind()'
    print(bound_args)  # create the 'BoundArguments' object
    for name, value in bound_args.arguments.items():
        print(name, '=', value)
    del my_tag['name']
    """bound_args = sig.bind(**my_tag)""" # won't work because lack default value for 'name'
    # function annotations
    def clip(text:str, max_len:'int > 0'=80,) -> str:
        pass
    print(clip.__annotations__)
    sig = signature(clip)
    print(sig.return_annotation)
    for param in sig.parameters.values():  # a parameters dict mapping name to objects
        note = repr(param.annotation).ljust(13)
        print(note, ':', param.name, '=', param.default)
    # operator module
    from functools import reduce
    def fact(n):
        return reduce(lambda a, b: a*b, range(1, n+1))
    from operator import mul
    def fact(n):
        return reduce(mul, range(1, n+1))  # operator replace the use of lambda function
    # itemgetter
    metro_data = [
        ('tokyo', 'jp', 36.933, (35.689722, 139.691667)),
        ('delhi', 'in', 21.935, (28.613889, 77.2088889)),
        ('mexico city', 'mx', 20.142, (19.433333, -99.133333)),
        ('new york-newark', 'us', 20.104, (40.808611, -74.020386)),
        ('sao paulo', 'br', 19.649, (-23.54778, -46.635833)),
    ]
    from operator import itemgetter
    for city in sorted(metro_data, key=itemgetter(1)):  # itemgetter performs like lambda fields: fields[1]
        print(city)
    cc_name = itemgetter(1, 0)  # get value at corresponding index
    for city in metro_data:
        print(cc_name(city))  # print only the selected values
    # attrgetter
    from collections import namedtuple
    LatLong = namedtuple('LatLong', 'lat long')
    Metropolis = namedtuple('Metropolis', 'name cc pop coord')
    metro_areas = [Metropolis(name, cc, pop, LatLong(lat, long))
                   for name, cc, pop, (lat, long) in metro_data]  # tuple listing and created nested tuple
    print(metro_areas[0])
    print(metro_areas[0].coord.lat)
    from operator import attrgetter
    name_lat = attrgetter('name', 'coord.lat')  # get arribute with the name
    for city in sorted(metro_areas, key=attrgetter('coord.lat')):
        print(name_lat(city))
    # methodcaller
    from operator import methodcaller
    s = 'the time has come'
    upcase = methodcaller('upper')
    print(upcase(s))
    print(str.upper(s))  # achive the same result
    hiphenate = methodcaller('replace', ' ', '-')
    print(hiphenate(s))
    # functools.partial
    from operator import mul
    from functools import partial
    triple = partial(mul, 3)  # binding positional argument to the function
    print(triple(7)) # function need only one more argument
    a = list(map(triple, range(1, 10)))
    print(a)
    import unicodedata, functools
    nfc = functools.partial(unicodedata.normalize, 'NFC')
    s1 = 'café'
    s2 = 'cafe\u0301'
    print(s1 == s2)
    print(nfc(s1) == nfc(s2))
    print(tag)
    picture = partial(tag, 'img', cls='pic-frame')
    print(picture(src='wumpus.jpeg'))
    print(picture)  # the partial function create a 'functools.partial' object, with access to original function and bound argument
    print(picture.func)
    print(picture.args)
    print(picture.keywords)
    


if __name__ == "__main__":
    main()