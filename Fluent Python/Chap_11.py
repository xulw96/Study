def main():
    # subclass an ABC
    import collections
    Card = collections.namedtuple('Card', ['rank', 'suit'])
    class FrenchDeck2(collections.MutableSequence):  # subclass of MutableSquence ABC
        ranks = [str(n) for n in range(2, 11)] + list('JQKA')
        suits = 'spades diamonds clubs hearts'.split()
        def __init__(self):
            self._cards = [Card(rank, suit) for suit in self.suits for ranks in self.ranks]
        def __len__(self):
            return len(self._cards)
        def __getitem__(self, position):
            return self._cards[position]
        def __setitem__(self, position, value):  # to enable shuffling
            self._cards[position] = value
        def __delitem__(self, position):  # needed to subclass
            del self._cards[position]
        def insert(self, position, value):  # needed to subclass
            self._cards.inserrt(position, value)
    import abc
    # create an ABC
    class Tombola(abc.ABC):
        @abc.abstractmethod
        def load(self, iterable):
            """Add items from an iterable."""
        @abc.abstractmethod
        def pick(self):
            """Remove item at random, returning it.
            This method should raise 'LookupError' when the instance is empty."""
        def loaded(self):
            """Return 'True' if there's at least 1 item, 'False' otherwise."""
            return bool(self.inspect())  # ABC can only rely on the interface of itself
        def inspect(self):
            """Return a sorted tuple with the items currently inside."""
            items = []
            while True:
                try:
                    items.append(self.pick())
                except LookupError:
                    break
            self.load(items)  # pick all items and then load back
            return tuple(sorted(items))
    # BingoCage
    import random
    class BingoCage(Tombola):  # shuffle and pop the last
        def __init__(self, items):
            self._randomizer = random.SystemRandom()
            self._items = []
            self.load(items)
        def load(self, items):
            self._items.extend(items)
            self._randomizer.shuffle(self._items)
        def pick(self):
            try:
                return self._items.pop()  # default last item
            except IndexError:
                raise LookupError('pick from empty BingoCage')
        def __call__(self):
            self.pick()
    # LotteryBlower
    class LotteryBlower(Tombola):  # randomly pop an item
        def __init__(self, iterable):
            self._balls = list(iterable)  # build a list from iterable, instead of holding a reference to the items
        def load(self, iterable):
            self._balls.extend(iterable)
        def pick(self):
            try:
                position = random.randrange(len(self._balls))
            except ValueError:  # change the Error message to be rasaised to fit in the intention
                raise LookupError('pick from empty LotteryBlower')
            return self._balls.pop(position)
        def loaded(self):  # override the ABC
            return bool(self._balls)
        def inspect(self):  # override the ABC
            return tuple(sorted(self._balls))
    # TomboList
    @Tombola.register  # register the class as a virtual subclass of 'Tombola' (not inherit methods)
    class TomboList(list):  # this is a real subclass of 'list' (not inherit methods)
        def pick(self):
            if self:  # a bool test of list
                position = randrange(len(self))
                return self.pop(position)
            else:
                raise LookupError('pop from empty TomboList')
        load = list.extend  # TomboList.load == list.extend
        def loaded(self):
            return bool(self)  # a bool(list) check whether it's empty
        def inspect(self):
            return tuple(self)
    # doctest
    """import doctest
    TEST_FILE = 'tombola_tests.rst'
    TEST_MSG = '{0:16} {1.attempted:2} tests, {1.failed:2} failed - {2}'
    def main(argv):
        verbose = '-v' in argv
        real_subclasses = Tombola.__subclasses__()  # this is a list of immediate real subclass
        virtual_subclasses = list(Tombola._abc_registry)  # build a list for concatenating. '_abc_registry' hold a weak reference to virtual subclass
        for cls in real_subclasses + virtual_subclasses:
            test(cls, verbose)
    def test(cls, verbose=False):
        res = doctest.testfile(
                TEST_FILE,
                globs={'ConcreteTombola':cls},  # bound the testing class to 'ConcreteTombola'
                verbose=verbose,
                optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)
        tag = 'FAIL' if res.failed else ' OK'
        print(TEST_MSG.format(cls.__name__, res, tag))
    if __name__ == "__main__":
        import sys
        main(sys.argv)"""


if __name__ == "__main__":
    main()