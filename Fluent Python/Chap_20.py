def main():
    # simple descriptor
    class Quantity:
        def __init__(self, storage_name):
            self.storage_name = storage_name
        def __set__(self, instance, value):  # self would be the descriptor instance, instance is the managed class instance
            if value > 0:
                instance.__dict__[self.storage_name] = value  # handle the __dict__ directly, to avoid calling __set__
            else:
                raise ValueError("Value must be > 0")
    class LineItem:
        weight = Quantity('weight')  # must give an input, or the class don't know which variable to bound with
        price = Quantity('price')
        def __init__(self, description, weight, price):
            self.description = description
            self.weight = weight
            self.price = price
        def subtotal(self):
            return self.weight * self.price

    # automatic storage_name updatign
    class Quantity:
        __counter = 0
        def __init__(self):
            cls = self.__class__  # a reference to this class
            prefix = cls.__name__
            index = cls.__counter
            self.storage_name = '_{}#{}'.format(prefix, index)  # an instance attribtue
        def __get__(self, instance, owner):  # self is the descriptor instance; instance is the managed instance, owner is the managed class
            if instance is None:
                return self
            else:
                return getattr(instance, self.storage_name)  # managed attribute is not the same name as storage_name
        def __set__(self, instance, value):
            if value > 0:
                setattr(instance, self.storage_name, value)  # the name is different, no need to use __dict__
    class LineItem:
        weight = Quantity()  # No need to provide attribute name
        price = Quantity()
        def __init__(self, description, weight, price):
            self.description = description
            self.weight = weight
            self.price = price
        def subtotal(self):
            return self.weight * self.price

    # same thing by property factory, not descriptor class
    def quantity():
        try:
            quantity.counter += 1
        except AttributeError:
            quantity.counter = 0
        storage_name = '_{}:{}'.format('quantity', quantity.name)
        def qty_getter(instance):
            return getattr(instance, storage_name)
        def qty_setter(instance, value):
            if value > 0:
                setattr(instance, storage_name, value)
            else:
                raise ValueError('value must be > 0')
        return property(qty_getter, qty_setter)

    # template method pattern
    import abc
    class AutoStorage:
        __counter = 0
        def __init__(self):
            cls = self.__class__
            prefix = cls.__name__
            index = cls.__counter
            self.storage_name = '_{}#{}'.format(prefix, index)
            cls.__counter += 1
        def __get__(self, instance, owner):
            if instance is None:
                return self
            else:
                return getattr(instance, self.storage_name)
        def __set__(self, instance, value):
            setattr(instance, self.storage_name, value)
    class Validated(abc.ABC, AutoStorage):  # abstract but also inherit
        def __set__(self, instance, value):
            value = self.validate(instance, value)  # delegating validation
            super().__set__(instance, value)  # use returned value to perform storage
        @abc.abstractmethod
        def validate(self, instance, value):
            """return validated value or raise ValueError"""
    class Quantity(Validated):
        """a number greater than zero"""
        def validate(self, instance, value):
            if value > 0:
                return value
            else:
                return ValueError('value must be > 0')
    class NonBlank(validated):
        """a string with a least one non-space character"""
        def validate(self, instance, value):
            value = value.strip()
            if len(value) == 0:
                raise ValueError('value cannont be empty or blank')
            else:
                return value


if __name__ == "__main__":
    main()
