def main():
    # sequence of words
    import re
    import reprlib
    RE_WORD = re.compile('\W+')
    class sentence:
        def __init__(self, text):
            self.text = text
            self.words = RE_WORD.findall(text)
        def __getitem__(self, index):  # when no __iter__ is found, __getitem__ is called from index 0
            return self.words[index]
        def __len__(self):
            return len(self.words)
        def __repr__(self):
            return 'Sentence(%s)' % reprlib.repr(self.text)
    # abc.Iterator class
    '''class Iterator(Iterable):
        __slots__ = ()
        @abstractmethod
        def __next__(self):
            'Return the next item from the iterator. When exhausted, raise StopIteration'
            raise StopIteration
        def __iter__(self):
            return self
        @classmethod
        def __subclasshook__(cls, c):  # for issubclass check
            if cls is Iterator:
                if (any("__next__" in B.__dict__ for B in c.__mro__) and any("__iter__" in B.__dict__ for B in c.__mro__)):
                    return True
            return NotImplemented'''
    # classic iterator
    class sentence:
        def __init__(self, text):
            self.text = text
            self.words = RE_WORD.findall(text)
        def __repr__(self):
            return 'Sentence(%s)' % reprlib.repr(self.text)
        def __iter__(self):
            return SentenceIterator(self.words)  # iterables return an iterator
    class SentenceIterator:
        def __init__(self, words):
            self.words = words
            self.index = 0  # used for iterating
        def __next__(self):  # handle the internal state of the iterator
            try:
                word = self.words[self.index]
            except IndexError:  # when the list becomes empty
                raise StopIteration()
            self.index += 1
            return word
        def __iter__(self):
            return self
    # generator function
    class sentence:
        def __init__(self, text):
            self.text = text
            self.words = RE_WORD.findall(text)
        def __repr__(self):
            return 'Sentence(%s)' % reprlib.repr(self.text)
        def __iter__(self):
            for word in self.words:
                yield word  # 'yield' is a signal for a generator: an iterator producing values passed to 'yield'
            return
    def gen_AB():
        print('start')
        yield 'A'  # this value will be consumed by 'for' loop and be assigned to variable 'c'
        print('continue')
        yield 'B'
        print('end.')
    print(gen_AB)  # a generator function
    print(gen_AB())  # returns a generator object: include the 'yield' word
    for c in gen_AB():  # the for machinery also fetches a generator object, and next() at each iteration
        print('-->', c)
    # lazy '__init__'
    class sentence:
        def __init__(self, text):  # not build a list of all words here.
            self.text = text
        def __repr__(self):
            return 'Sentence(%s)' % reprlib.repr(self.text)
        def __iter__(self):
            for match in RE_WORD.finditer(self.text):  # 're.finditer' returns a generator object that only produce the next word when needed, thus saving memory
                yield match.group()  # extract the actual matched text
    # generator expression
    class sentence:
        def __init__(self, text):
            self.text = text
        def __repr__(self):
            return 'Sentence(%s)' % reprlib.repr(self.text)
        def __iter__(self):
            return (match.group() for match in RE_WORD.finditer(self.text))  # return a built generator (better than listcomps to save memory)
    # independent (of data source) generator
    class ArithmeticProgression:
        def __init__(self, begin, step, end=None):
            self.begin = begin
            self.step = step
            self.end = end
        def __iter__(self):
            result = type(self.begin + self.step)(self.begin)  # the value is self.begin, the type is coerced by addition operator
            forever = self.end is None
            index = 0
            while forever or result < self.end:
                yield result  # produce the current value
                index += 1
                result = self.begin + self.step * index
    # with itertools
    import itertools
    def aritprog_gen(begin, step, end=None):
        first = type(begin + step)(begin)
        ap_gen = itertools.count(first, step)  #
        if end is not None:
            ap_gen = itertools.takewhile(lambda n: n < end, ap_gen)  # take while consumes another generator and iterates within a give range.




if __name__ == "__main__":
    main()
