def main():
    # build dict
    a = dict(one=1, two=2, three=3)
    b = {'one': 1, 'two': 2, 'three': 3}
    c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
    d = dict([('two', 2), ('one', 1), ('three', 3)])
    e = dict({'three': 3, 'one': 1, 'two': 2})
    print(a == b == c == d == e)
    # dictcomp
    DIAL_CODES = [
        (86, 'china'),
        (91, 'india'),
        (1, 'united state'),
        (62, 'indonesia'),
        (55, 'brazil'),
        (92, 'pakistan'),
        (880, 'bangladesh'),
        (234, 'nigeria'),
        (7, 'russia'),
        (81, 'japan'),
    ]
    country_code = {country: code for code, country in DIAL_CODES}
    country_code_2 = {code: country.upper() for country, code in country_code.items() if code < 66}
    print(country_code)
    print(country_code_2)
    # handle missing keys by 'setdefault'
    """# key is word; value is its occurences
    import sys
    import re
    WORD_RE = re.compile('\w+')
    index = {}
    with open(sys.argv[1], encoding='utf-8') as fp:
        for line_no, line in enumerate(fp, 1):
            for match in WORD_RE.finditer(line):
                word = match.group()
                column_no = match.start() + 1
                location = (line_no, column_no)
                occurences = index.get(word, [])
                occurences.append(location)
                index[word] = occurences
                index.setdefault(word, []).append(location)  # same method for above and unter
                if word not in index:
                    index[word] = []
                index[word].append(location)
    for word in sorted(index, key=str.upper):  # use key to sort the dict
        print(word, index[word])
    # handle missing keys by 'defaultdict'
    import sys
    import re
    import collections
    WORD_RE = re.compile('\w+')
    index = collections.defaultdict(list)  # not an empty list, but a defaultdict
    with open(sys.argv[1], encoding='utf-8') as fp:
        for line_no, line in enumerate(fp, 1):
            for match in WORD_RE.finditer(line):
                word = match.group()
                column_no = match.start() + 1
                location = (line_no, column_no)
                index[word].append(location)  # defaultdict provide value for __getitem__ calls, not other methods (eg, __contains__)
    for word in sorted(index, key=str.upper):
        print(word, index[word])"""
    # StrKeyDict, convert to str when number can not be found in key
    class StrKeyDict0(dict):  # build subclass for dict

        def __missing__(self, key):
            if isinstance(key, str):  # this test is crucial, or self[str(key)] will end up in recursion.
                raise KeyError(key)
            return self[str(key)]

        def get(self, key, default=None):  # actually '__getitem__'
            try:
                return self[key]
            except KeyError:
                return default

        def __contains__(self, key):  # works for the 'in' operator
            return key in self.keys() or str(key) in self.keys()  # self.keys() is more efficient than self(must scan the whole list)

    d = StrKeyDict0([('2', 'two'), ('3', 'three')])
    print(d['2'])
    print(d[3])
    print(d.get('2'))
    print(d.get(3))
    print('2' in d)
    print(3 in d)  # works for both 'd[]', '.get()' and 'in' operator.
    # counter
    import collections
    ct = collections.Counter('Deutschland')
    print(ct)
    ct.update('Preussen')
    print(ct)
    print(ct.most_common(2))
    # UserDict
    import collections
    class StrKeyDict(collections.UserDict):
        def __missing__(self, key):
            if isinstance(key, str):
                raise KeyError(key)
            return self[str(key)]

        def __contains__(self, key):
            return str(key) in self.data  # userdict use internal dict ('data') instead of inheriting from dict

        def __setitem__(self, key, item):
            self.data[str(key)] = item
    # MappingProxy
    from types import MappingProxyType
    d = {1: 'A'}
    d_proxy = MappingProxyType(d)
    print("d_proxy:", d_proxy)
    print(d_proxy[1])
    # d_proxy[2] = proxytype is immutable
    d[2] = 'x'
    print(d_proxy[2])  # change in d is reflected in proxy
    # set
    l = ['spam', 'spam', 'egg', 'spam']
    print(set(l))  # set has unique objects, can remove duplication
    print(list(set(l)))
    # set operator
    """found = len(needles & haystack)  # set operator: '&' for intersection, '|' for union, '-' for difference
    found = 0  # same task accomplished by coding
    for n in needles:
        if n in haystack:
            found += 1"""
    # build set
    """found = len(set(needles) & set(haystack))
    found = len(set(needles).intersection(haystack))"""  # another way
    s = {1}
    print(type(s))
    print(s)
    print(s.pop())  # return and remove the first item
    print(s)  # set() for an empty set
    print(frozenset(range(10)))
    from unicodedata import name
    set_comps = {chr(i) for i in range(32, 256) if 'SIGN' in name(chr(i), '')}
    print(set_comps)
    # Hash
    DIAL_CODES = [
        (86, 'china'),
        (91, 'india'),
        (1, 'united states'),
        (62, 'indonesia'),
        (55, 'brazil'),
        (92, 'pakistan'),
        (889, 'bangladesh'),
        (234, 'nigeria'),
        (7, 'russia'),
        (81, 'japan'),
    ]
    d1 = dict(DIAL_CODES)
    d2 = dict(sorted(DIAL_CODES))
    d3 = dict(sorted(DIAL_CODES, key=lambda x: x[1]))
    print(d1.keys())
    print(d2.keys())
    print(d3.keys())
    print(d1 == d2 and d2 == d3)


if __name__ == "__main__":
    main()