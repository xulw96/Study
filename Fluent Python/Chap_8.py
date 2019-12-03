def main():
    # shallow copy
    l1 = [3, [66, 55, 44], (7, 8, 9)]
    l2 = list(l1)  # create a shallow copy of l1
    '''l2 = l1[:]'''  # same as above
    print(l1 == l2)  # same value
    print(l1 is l2)  # different identity
    l1.append(100)  # no effect on l2
    l1[1].remove(55)  # effect on l2
    print('l1:', l1)
    print('l2:', l2)
    l2[1] += [33, 22]  # affect l1, change in place
    l2[2] += (10, 11)  # new tuple will be created, and l1[2] will be different object with l2[2]
    print('l1:', l1)
    print('l2:', l2)

    # copy
    class Bus:
        def __init__(self, passengers=None):
            if passengers is None:
                self.passengers = []
            else:
                self.passengers = list(passengers)  # a new list (shallow copy to each value, not the whole) with the value, not being assigned as an alias
        def pick(self, name):
            self.passengers.append(name)
        def drop(self, name):
            self.passengers.remove(name)
    import copy
    bus1 = Bus(['Alice', 'Bill', 'Claire', 'David'])
    bus2 = copy.copy(bus1)
    bus3 = copy.deepcopy(bus1)
    print('id for bus1, bus2, and bus3:', [id(bus1), id(bus2), id(bus3)])
    print('id for bus.passengers:', [id(bus1.passengers), id(bus2.passengers), id(bus3.passengers)])  # the pointer place
    bus1.drop('Bill')
    print(bus2.passengers, bus3.passengers)

    # cyclic reference
    a = [10, 20]
    b = [a, 30]
    a.append((b))  # a cyclic reference
    print(a)
    c = copy.deepcopy(a)  # still manages to copy value of a
    print(c)

    # parameter sharing
    def f(a, b):
        a += b
        return a
    a = [1, 2]
    b = [3, 4]
    print(f(a, b))
    print(a)  # the list is changed in place
    a = (1, 2)
    b = (3, 4)
    print(f(a, b))
    print(a)  # the tuple is unchanged, but new one was created

    # mutable default is a danger
    class HauntedBus:
        '''A bus model huanted by ghost passengers'''
        def __init__(self, passengers=[]):  # a mutable default value
            self.passengers = passengers  # 'self.passengers' is only an alias to the empty list
        def pick(self, name):
            self.passengers.append(name)
        def drop(self, name):
            self.passengers.remove(name)
    bus1 = HauntedBus(['Alice', 'Bill'])
    bus1.pick('charlie')
    print(bus1.passengers)  # works fine when input is not empty
    bus2 = HauntedBus()
    bus2.pick('carrie')
    bus3 = HauntedBus()
    print(bus2.passengers, bus3.passengers)  # bus2 and bus3 both have alias to the empty list

    # mutable default is a danger (2)
    class TwilightedBus:
        def __init__(self, passengers=None):
            if passengers is None:
                self.passengers = []
            else:
                self.passengers = passengers  # assign the input list as an alias
        def pick(self, name):
            self.passengers.append(name)
        def drop(self, name):
            self.passengers.remove(name)
    basketball_team = ['sue', 'tina', 'maya', 'diana', 'pat']
    bus = TwilightedBus(basketball_team)
    bus.drop('tina')
    print(basketball_team)  # the original list is changed by the call

    # end of an object
    import weakref
    s1 = {1, 2, 3}
    s2 = s1  # both refer to the same list
    def bye():
        print("gone with the wind")
    ender = weakref.finalize(s1, bye)  # call the function when the object are garbage collected
    del s1  # delete the reference, not the object
    print(ender.alive)
    s2 = 'spam'  # the object becomes unreachable, 'bye' is called upon
    print(ender.alive)

    # weak reference
    a = {0, 1}
    wref = weakref.ref(a)  # create a weak reference to 'a'
    a = {1, 2}
    print(wref())  # reference count is not increased
    b = [1, 2]
    '''wref = weakref.ref(b)'''  # 'list' and 'dict' can't be weak referenced
    class Mylist(list):
        '''list subclass whose instances may be weakly refenreced'''
    a_list = Mylist(range(10))
    wref = weakref.ref(a_list)

    # WeakValueDictionary
    class Cheese:
        def __init__(self, kind):
            self.kind = kind
        def __repr__(self):
            return 'Cheese(%r)' % self.kind
    stock = weakref.WeakValueDictionary()
    catalog = [Cheese('Red Leicester'), Cheese('Tilsit'), Cheese('Brie'), Cheese('Parmesan')]
    for cheese in catalog:
        stock[cheese.kind] = cheese  # the name is mapped as a weak reference to the catalog
    print(sorted(stock.keys()))
    del catalog  # losing value will lose the keys
    print(sorted(stock.keys()))  # not empty because the for loop hold a reference to cheese
    del cheese
    print(sorted(stock.keys()))


if __name__ == "__main__":
    main()