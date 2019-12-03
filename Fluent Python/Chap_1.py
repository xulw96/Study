def main():
    # grant attributes instead of methods(like a database)
    import collections
    Card = collections.namedtuple('card', ['rank', 'suit'])

    # making an object and passing on core features
    class FrenchDeck:
        ranks = [str(n) for n in range(2, 11)] + list('JQKA')
        suits = 'spades diamonds clubs hearts'.split()  # create a list with the four items

        def __init__(self):
            self._cards = [Card(rank, suit) for rank in self.ranks for suit in self.suits]

        def __len__(self):
            return len(self._cards)

        def __getitem__(self, position):
            return self._cards[position]
    # single item from __init__
    beer_card = Card('7', 'diamonds')
    print("info of beer_card is:", beer_card)
    # length from __len__
    deck = FrenchDeck()
    print("length of deck is:", len(deck))
    # slicing from __getitem__
    print("deck[0] is :", deck[0])
    print("deck[-1] is :", deck[-1])
    # random selection
    from random import choice
    print("choose a card randomly:", choice(deck))
    print("choose a card randomly:", choice(deck))
    # using slicing with [] operator
    print("top three cards are:", deck[:3])
    print("start on 12th, skipping 13 cards at one time:\n", deck[12::13])
    # iteration
    for card in deck:
        print(card)
    for card in reversed(deck):
        print(card)
    print(Card('Q', 'hearts') in deck)
    print(Card('7', 'beasts') in deck)
    # sorting
    suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)  # works like a mapping table

    def spades_high(card):
        rank_value = FrenchDeck.ranks.index(card.rank)  # give both numbers and alphabets an index number
        return rank_value * len(suit_values) + suit_values[card.suit]  # passing the mapping value
    for card in sorted(deck, key=spades_high):  # sorting table by the key value
        print(card)
    # vector
    from math import hypot

    class Vector:
        def __init__(self, x=0, y=0):
            self.x = x
            self.y = y

        def __repr__(self):
            return 'Vector(%r, %r)' % (self.x, self.y)

        def __abs__(self):
            return hypot(self.x, self.y)

        def __bool__(self):
            return bool(abs(self))

        def __add__(self, other):
            x = self.x + other.x
            y = self.y + other.y
            return Vector(x, y)

        def __mul__(self, scalar):
            return Vector(self.x * scalar, self.y * scalar)


if __name__ == "__main__":
    main()
