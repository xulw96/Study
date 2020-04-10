def main():
    # vector2d class
    from array import array
    import math
    class Vector2d:
        """__slots__ = ('__x', '__y')"""  # saving memory by store instance attributes in tuple, not dict
        typecode = 'd'  # used when converting to/from bytes. This class attribute can be directly overriden
        def __init__(self, x, y):
            self.__x = float(x)  # a privaet attribute
            self.__y = float(y)
        @property  # the .x and .y become read-only (immutable)
        def x(self):
            return self.__x
        @property
        def y(self):
            return self.__y
        def __iter__(self):  # enable unpacking to work
            return (i for i in (self.x, self.y))
        def __repr__(self):
            class_name = type(self).__name__  # help for user when creating subclass of Vector2d
            return '{}({!r}, {!r})'.format(class_name, *self)  # '*self' feed x and y to format
        def __str__(self):
            return str(tuple(self))  # build a tuple for display
        def __bytes__(self):
            return (bytes([ord(self.typecode)]) + bytes(array(self.typecode, self)))  # typecode to bytes and concatenate array built by iterating over instances
        def __eq__(self, other):  # also required to become hashable
            return tuple(self) == tuple(other)  # build tuple to compare all components
        def __hash__(self):
            return hash(self.x) ^ hash(self.y)  # '^' operator just mix the hashes
        def __abs__(self):
            return math.hypot(self.x, self.y)
        def __bool__(self):
            return bool(abs(self))  # nonzero is True
        @classmethod  # to define a method on class, not on instances
        def frombytes(cls, octets):  # the class itself is passed into the function
            typecode = chr(octets[0])
            memv = memoryview(octets[1:].cast(typecode))
            return cls(*memv)  # invoke the class to build an instance
        def angle(self):
            return math.atan2(self.y, self.x)  # get the angle
        def __format__(self, fmt_spec=''):
            if fmt_spec.endswith('p'):  # employ a transformation to (magnitude, angle)
                fmt_spec = fmt_spec[:-1]  # remove the 'p'
                coords = (abs(self), self.angle())
                outer_fmt = '<{}, {}>'
            else:
                coords = self
                outer_fmt = '({}, {})'
            components = (format(c, fmt_spec) for c in coords)  # use 'format()' to apply format specifier
            return outer_fmt.format(*components)  # sustain the formula


    # classmethod and staticmethod
    class Demo:
        @classmethod
        def klassmeth(*args):
            return args
        @staticmethod
        def statmeth(*args):
            return args
    print(Demo.klassmeth('spam'))  # get the class into argument
    print(Demo.statmeth())  # not get the class as argument

    # formatting
    brl = 1/2.43
    form = format(brl, '0.4f')  # using format() to change the string's format
    print(form)
    form = '1 BRL = {rate:0.2f} USD'.format(rate=brl)  # use .format(). Left of colon is the field name; right of it is the formating specifier
    print(format(Vector2d(1, 1), '0.3fp'))

    # name mangling
    v1 = Vector2d(3, 4)
    print(v1.__dict__)  # the private attribtue is stored in '__dict__'
    print(v1._Vector2d__x)

    # class attribute overiding
    v1 = Vector2d(1.1, 2.2)
    v1.typecode = 'f'  # directly change
    class ShortVector2d(Vector2d):  # change by a subclass
        typecode = 'f'


if __name__ == "__main__":
    main()
