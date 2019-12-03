def main():
    # Order class with discount strategies
    from abc import ABC, abstractmethod
    from collections import namedtuple
    Customer = namedtuple('Customer', 'name fidelity')
    class LineItem:
        def __init__(self, product, quantity, price):
            self.product = product
            self.quantity = quantity
            self.price = price
        def total(self):
            return self.price * self.quantity
    class Order:
        def __init__(self, customer, cart, promotion=None):
            self.customer = customer
            self.cart = list(cart)
            self.promotion = promotion
        def total(self):
            if not hasattr(self, '__total'):
                self.__total = sum(item.total() for item in self.cart)
            return self.__total
        def due(self):
            if self.promotion is None:
                discount = 0
            else:
                discount = self.promotion.discount(self)
            return self.total() - discount
        def __repr__(self):
            fmt = '<Order total: {:.2f} due: {:2f}>'
            return fmt.format(self.total(), self.due())

    class Promotion(ABC):  # Promotion as an abstract base class
        @abstractmethod
        def discount(self, order):
            """Return discount as a positive dollar amount"""
    class FidelityPromo(Promotion):
        """5% discount for customers with 1000 or more fidelity points"""
        def discount(self, order):
            return order.total() * 0.5 if order.customer.fidelity >= 1000 else 0
    class BulkItemPromo(Promotion):
        """ 10% discount for each LineItem with 20 or more units"""
        def discount(self, order):
            discount = 0
            for item in order.cart:
                if item.quantity >= 20:
                    discount += item.total() * 0.1
            return discount
    class LargeOrderPromo(Promotion):
        """7% discount for orders with 10 or more distinct items"""
        def discount(self, order):
            distinct_items = {item.product for item in order.cart}
            if len(distinct_items) >= 10:
                return order.total() * 0.07
            return 0
    # test the function
    joe = Customer('John Doe', 0)
    ann = Customer('Ann Smith', 1100)
    cart = [LineItem('banana', 4, 0.5),
            LineItem('apple', 10, 1.5),
            LineItem('watermellon', 5, 5.0)]
    print(Order(joe, cart, FidelityPromo()), Order(ann, cart, FidelityPromo()), sep='\n')  # call the __repr__ in Order class
    banana_cart = [LineItem('banana', 30, 0.5),
                   LineItem('apple', 10, 1.5)]
    print(Order(joe, banana_cart, BulkItemPromo()))
    long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]
    print(Order(joe, long_order, LargeOrderPromo()), Order(joe, cart, LargeOrderPromo()), sep='\n')
    # function-oriented strategy
    from collections import namedtuple
    Customer = namedtuple('Customer', 'name fidelity')
    class LineItem:
        def __init__(self, product, quantity, price):
            self.product = product
            self.quantity = quantity
            self.price = price
        def total(self):
            return self.price * self.quantity
    class Order:
        def __init__(self, customer, cart, promotion=None):
            self.customer = customer
            self.cart = list(cart)
            self.promotion = promotion
        def total(self):
            if not hasattr(self, '__total'):
                self.__total = sum(item.total() for item in self.cart)
            return self.__total
        def due(self):
            if self.promotion is None:
                discount = 0
            else:
                discount = self.promotion(self)  # directly call the self.promotion function. No abstract class
            return self.total() - discount
        def __repr__(self):
            fmt = '<Order total: {:.2f} due: {:2f}>'
            return fmt.format(self.total(), self.due())
    def fidelity_promo(order):  # put each strategy as a function
        """5% discount for customers with 1000 or more fidelity points"""
        return order.total() * 0.5 if order.customer.fidelity >= 1000 else 0
    def bulk_item_promo(order):
        """10% discount for each LineItem with 20 or more units"""
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * 0.1
        return discount
    def large_order_promo(order):
        """ 7% discount for orders with 10 or more distinct items"""
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * 0.07
        return 0
    # test the above function
    joe = Customer('John Doe', 0)
    ann = Customer('Ann Smith', 1100)
    cart = [LineItem('banana', 4, 0.5),
            LineItem('apple', 10, 1.5),
            LineItem('watermellon', 5, 5.0)]
    print(Order(joe, cart, fidelity_promo), Order(ann, cart, fidelity_promo),
          sep='\n')  # call the __repr__ in Order class
    banana_cart = [LineItem('banana', 30, 0.5),
                   LineItem('apple', 10, 1.5)]
    print(Order(joe, banana_cart, bulk_item_promo)) # the function doesn't need input ('()') like class
    long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]
    print(Order(joe, long_order, large_order_promo), Order(joe, cart, large_order_promo), sep='\n')
    # find the best_promo
    promos = [fidelity_promo, bulk_item_promo, large_order_promo]
    def best_promo(order):
        """select best discount available"""
        return max(promo(order) for promo in promos)
    print(Order(joe, long_order, best_promo),
          Order(joe, banana_cart, best_promo),
          Order(ann, cart, best_promo), sep='\n')
    # 'globals()' and 'inspect.getmembers()'
    promos = [globals()[name] for name in globals()  # iterate over each name in dict
              if name.endswith('_promo')
              and name != 'best_promo']
    # module inspect
    import inspect
    promos = [func for name, func in inspect.getmembers(Promotion, inspect.isfunction)]
    # MacroCommand class with list of commans
    class MacroCommand:
        """A command that executes a list of commands"""
        def __init__(self, commands):
            self.commands = list(commands)  # a locap copy: list of command functions
        def __call__(self):  # enable this class to be callable
            for command in self.commands:
                command()



if __name__ == "__main__":
    main()