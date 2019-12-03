def main():
    # Version of Vector
    from array import array
    import reprlib
    import math
    import functools
    import operator
    import itertools
    class Vector:
        typecode = 'd'
        def __init__(self, components):
            self._components = array(self.typecode, components)  # a 'protected' attribute
        def __iter__(self):
            return iter(self._components)
        def __repr__(self):
            components = reprlib.repr(self._components)  # a limited length representation. A constructor to show the vector
            components = components[components.find('['):-1]  # return the index for the range between '[' and last one. (this can be achieved by list repr)
            return 'Vector({})'.format(components)
        def __bytes__(self):
            return (bytes([ord(self.typecode)]) + bytes(self._components))
        def __eq__(self, other):
            if len(self) != len(other):
                return False
            for a, b in zip(self, other):
                if a != b:
                    return False
            return True
            # return len(self) == len(other) and all(a == b for a, b in zip(self, other)), same as above with function 'all'
            # return tuple(self) == tuple(other), not efficient enough
        def __hash__(self):
            hashes = (hash(x) for x in self._components)
            # hashes = map(hash, self._components), work the same as above
            return functools.reduce(operator.xor, hashes, 0)  # last value as initializer
        def __abs__(self):
            return math.sqrt(sum(x * x for x in self))  # can't use hypot for over-2 dimension
        def __bool__(self):
            return bool(abs(self))
        def __len__(self):  # __len__ and __getitem__ are necessary methods for a sequence class
            return len(self._components)
        def __getitem__(self, index):
            cls = type(self)
            if isinstance(index, slice):  # check the argument to be a 'slice' instance
                return cls(self._components[index])  # another Vector instance from the slice array
            elif isinstance(index, numbers.Integral):
                return self._components[index]  # return the specific item
            else:
                msg = '{cls.__name__} indices must be integers'
                raise TypeError(msg.format(cls=cls))
        shortcut_names = 'xyzt'
        def __getattr__(self, name):  # enable using shortcut to accessing value
            cls = type(self)
            if len(name) == 1:
                pos = cls.shortcut_names.find(name)  # find will return the index for target
                if 0 <= pos < len(self._components):
                    return self._components
            msg = '{.__name__!r} object has no attribute {!r}'
            raise AttributeError(msg.format(cls,name))
        def __setattr__(self, name, value):  # avoid shorcut value being changed as attribute
            cls = type(self)
            if len(name) == 1:
                if name in cls.shortcut_names:
                    error = 'readonly attribute {attr_name!r}'
                elif name.islower():
                    error = "can't set attributes 'a' to 'z' in {cls_name!r}'"
                else:
                    error = ''
            if error:
                msg = error.format(cls_name=cls.__name__, attr_name=name)
                raise AttributeError(msg)
            super.__setattr__(name, value)  # enable changing the attribute at superclass
        def angle(self, n):
            r = math.sqrt(sum(x * x for x in self[n:]))
            a = math.atan2(r, self[n-1])
            if (n == len(self) - 1) and (self[-1] < 0):
                return math.pi * 2 - a
            else:
                return a
        def angles(self):  # build a generator on the angle function
            return (self.angle(n) for n in range(1, len(self)))
        def __format__(self, fmt_spec):
            if fmt_spec.endswith('h'):  # 'h' for hyperspherical coordinates
                fmt_spec = fmt_spec[:-1]
                coords = itertools.chain([abs(self)], self.angles)  # 'itertools.chain' produce a genexp
                outer_fmt = '<{}>'
            else:
                coords = self
                outer_fmt = '({})'
            components = (format(c, fmt_spec) for c in coords)
            return outer_fmt.format(','.join(components))
        @classmethod
        def frombytes(cls, octets):
            typecode = chr(octets[0])
            memv = memoryview(octets[1:].cast(typecode))
            return cls(memv)  # no unpack with '*'

if __name__ == "__main__":
    main()