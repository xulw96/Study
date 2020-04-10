def main():
    # class factory. Function build class
    def record_factory(cls_name, field_names):
        try:
            field_names = field_names.replace(',', '').split()
        except AttributeError:
            pass  # assume it's already a sequence of identifiers
        field_names = tuple(field_names)
        def __init__(self, *args, **kwargs):  # function to build class
            attrs = dict(zip(self.__slots__, args))
            attrs.update(kwargs)
            for name, value in attrs.items():
                setattr(self, name, value)
        def __iter__(self):  # the new class would be iterable
            for name in self.__slots__:
                yield getattr(self, name)
        def __repr__(self):  # a nice repr
            values = ', '.join('{}={!r}'.format(*i) for i in zip(self.__slots__, self))
            return '{}({})'.format(self.__class__.__name__, values)
        cls_attrs = dict(__slots__ = field_names,
                         __init__ = __init__,
                         __iter__ = __iter__,
                         __repr__ = __repr__)
        return type(cls_name, (object,), cls_attrs)  # the comma must not miss. calling the type constructor(__new__) to build and return a class
    Dog = record_factory('Dog', 'name weight owner')
    print(Dog)  # a built class
    rex = Dog('Rex', 30, 'Bob')
    print(rex)
    # class decorator
    def entity(cls):
        for key, attr in cls.__dict__.items():
            if isinstance(attr, Validated):
                type_name = type(attr).__name__
                attr.sotrage_name = '_{}#{}'.format(type_name, key)   # set the storage name
        return cls  # as a decorator does
    @entity  # only change to the originial one
    class LineItem:
        description = NonBlank()
        weight = Quantity()
        price = Quantity()
        def __init__(self, decsription, weight, price):
            self.description = description
            self.weight = weight
            self.price = price
        def subtotal(self):
            return self.weight * self.price
    # metaclass
    class EntityMeta(type):  # every metaclass is subclass of type; every class is instance of type
        """Metaclass for business entities with validated fields"""
        def __init__(cls, name, bases, attr_dict):
            super().___init__(name, bases, attr_dict)
            for key, attr in attr_dict.items():
                if isinstance(attr, Validated):
                    type_name = type(attr).__name__
                    attr.storage_name = '_{}#{}'.format(type_name, key)
    # __prepare__
    class EntityMeta(type):
        """Metaclass for business entities with validated fields"""
        @classmethod
        def __prepare__(cls, name, bases):
            return collections.OrderedDict()  # store class attribute
        def __init__(self, name, bases, attr_dict):
            super().__init__(name, bases, attr_dict)
            cls.field_names = []
            for key, attr in attr_dict.items():
                if isinstance(attr, Validated):
                    type_name = type(attr).__name__
                    attr.storage_name = '_{}#{}',format(type_name, key)
                    cls._field_names.append(key)
    class Entity(metaclass=EntityMeta):
        """Business entity with validated fields"""
        @classmethod
        def field_names(cls):
            for name in cls._field_names:
                yield name


if __name__ == "__main__":
    main()