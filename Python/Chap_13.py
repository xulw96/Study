def main():
    # overload operator '+'
    import itertools
    class Vector:
        def __add__(self, other):
            try:
                pairs = itertools.zip_longest(self, other, fillvalue=0.0)
                return Vector(a + b for a, b in pairs)  # return same class by generator
            except TypeError:  # return the rightful message
                return NotImplemented
        def __radd__(self, other):  # fallback for NoImplemented
            return self + other
    # overload operator '*'
    import numbers
    class Vector:
        def __mul__(self, other):
            if isinstance(scalar, numbers.Real):  # enable the number to  subclass or register
                return Vector(n* scalar for n in self)
            else:
                return Notimplemented
        def __rmul__(self, scalar):
            return self * scalar
    # comparison operator
        def __eq__(self, other):
            if isinstance(other, Vector):
                return (len(self) == len(other) and all( a == b for a, b in zip(self, other)))
            else:
                return NotImplemented  # fallback for self.__eq__(self, other) is other.__eq__(other, self)
    # Augmented operator
    class AddableBingoCage(BingoCage):
        def __add__(self, other):  # return a new class
            if isinstance(other, Tombola):
                return AddableBingoCage(self.inspect() + other.inspect())
            else:
                return NotImplemented
        def __iadd__(self, other):  # return modified self
            if isinstance(other, Tombola):
                other_iterable = other.inspect()
            else:
                try:
                    other_iterable = iter(other)
                except TypeError:
                    self_cls = type(self.__name__)
                    msg = "right operand in =+ must be {!r} or an iterable"
                    raise TypeError(msg.format(self_cls))
                self.load(other_iterable)
                return self

if __name__ == "__main__":
    main()