def main():
    # tuple
    tup = tuple('string')
    print(tup)
    nested_tup = (4, 5, 6), (7, 8)
    print(nested_tup)
    con_tup = (4, None, 'foo') + (6, 0) + ('bar',)  # the comma is needed to change it to tuple instead of string
    print(con_tup)  # objects not copied, only reference to them
    seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    for a, b, c in seq:
        print("a={0}, b={1}, c={2}".format(a, b, c))
    values = 1, 2, 3, 4, 5, 5
    a, b, *_ = values
    print(*_)
    print(values.count(5))
    # list
    """everything = []
    for chunk in list_lists:
        everything.extend(chunk)  
        everthing = everthing + chunk"""  # slower: require create new list and object copy over
    # sequence functions
    """i = 0
    for value in collection:
        i += 1
    for i, value in enumerate(collection):""" # show the meaning of 'enumerate

    some_list = ['foo', 'bar', 'baz']
    mapping = {}
    for i, v in enumerate(some_list):  # creating indedx(key) to values
        mapping[v] = i
    print(mapping)

    seq1 = ['foo', 'bar', 'baz']
    seq2 = ['one', 'two', 'three']
    seq3 = ['False', 'True']
    zip_list = list(zip(seq1, seq2, seq3))  # pair up under limit of shorted list
    print(zip_list)
    for i, (a, b) in enumerate(zip(seq1, seq2)):  # combine 'enumerate' with 'zip'
        print('{0}: {1}, {2}'.format(i, a, b))

    pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')]
    first_name, last_name = zip(*pitchers)  # this convert row into column
    print(first_name)
    print(last_name)

    # dict
    words = ['apple', 'bat', 'bar', 'atom', 'book']
    by_letter = {}
    for word in words:
        letter = word[0]
        if letter not in by_letter:
            by_letter[letter] = [word]
        else:
            by_letter[letter].append(word)  # append instead of update
    print(by_letter)
    del by_letter
    by_letter = {}
    for word in words:
        letter = word[0]
        by_letter.setdefault(letter, []).append(word)  # use 'setdefault' to handle missing value
    print(by_letter)
    from collections import defaultdict  # use 'defaultdict' to handle missing value
    by_letter = defaultdict(list)
    for word in words:
        by_letter[word[0]].append(word)
    print(by_letter)
    # hashability
    d = {}
    d[tuple([1, 2, 3])] = 5  # any hashable object can be key
    print(d)
    # comps
    strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
    list_comp = [x.upper() for x in strings if len(x) > 2]  # a conditional list
    print(list_comp)
    unique_lengths = {len(x) for x in strings}  # a set
    print(unique_lengths)
    loc_mapping = {index: val for index, val in enumerate(strings)}  # enumerate a dict
    print(loc_mapping)
    names_of_interest = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
                         ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
    result = [name for names in names_of_interest for name in names if name.count('e') >= 2]  # nested comps
    print(result)
    some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    flattened = [x for tup in some_tuples for x in tup]  # flatten a list by nested comps
    print(flattened)
    list_of_lists = [[x for x in tup] for tup in some_tuples]  # note the order of comps is changed now
    print(list_of_lists)
    # Namespace
    def func():
        a = []  # local variable will be destroyed after this function
        for i in range(5):
            a.append(i)
    a = []  # variable defined outside function is considered global
    def func():
        for i in range(5):
            a.append(i)
    # return values from function
    def f():
        a = 5
        b = 6
        c = 7
        return a, b, c
    return_value = f()  # return a tuple
    a, b, c = return_value  # tuple unpacking
    a, b, c = f()  # this can replace the last two lines
    # function as object
    states = [' Alabama ', 'Georgia!', 'geogia', 'FlOrIda', 'south   carolina##', 'West virginia?']
    import re
    def clean_strings(strings):
        result = []
        for value in strings:
            value = value.strip()
            value = re.sub('[!#?]', '', value)
            value = value.title()
            result.append(value)
        return result
    print(clean_strings(states))
    # list of functions
    def remove_punctuation(value):
        return re.sub('[!#?]', '', value)
    clean_ops = [str.strip, remove_punctuation, str.title]
    def clean_strings(strings, ops):
        result = []
        for value in strings:
            for function in ops:  # run the function in clean_ops
                value = function(value)
            result.append(value)
        return result
    print(clean_strings(states, clean_ops))
    # functions as arguments
    for x in map(remove_punctuation, states):  # apply functions to sequence.
        print(x)
    # lamba function
    strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
    strings.sort(key=lambda x: len(set(list(x))))  # function as argument
    print(strings)

if __name__ == "__main__":
    main()