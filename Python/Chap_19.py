def main():
    # download the file
    from urllib.request import urlopen
    import warnings
    import os
    import json
    URL = 'http://www.oreilly.com/pub/sc/osconfeed'
    JSON = 'JSON'
    JSON_file = os.path.join(os.path.dirname(__file__), 'JSON')
    def load():
        if not os.path.exists(JSON):
            msg = 'downloading {} to {}'.format(URL, JSON)
            warnings.warn(msg)
            with urlopen(URL) as remote, open(JSON_file, 'wb') as local:  # two context managers to read and save
                local.write(remote.read())
        with open(JSON_file, encoding="utf-8") as fp:
            return json.load(fp)  # return native python objects
    load()
    feed = load()  # a python dict
    # a new class to hold the JSON file
    from collections import abc
    import keyword
    class FrozenJson:
        """A read-only facade for navigating a JSON-like obejct
        using attribute notation
        """
        def __init__(self, mapping):
            self.__data = {}  # get a dict; get a local copy
            for key, value in mapping.items():
                if keyword.iskeyword(key):  # plus a '_' incase it's a python reserved keyword
                    key += '_'
                self.__data[key] = value
        def __getattr__(self, name):
            if hasattr(self.__data, name):
                return getattr(self.__data, name)
            else:
                return FrozenJson.build(self.__data[name])  # transfrom from '[]' to '.()'.Navigating nested structures
        @classmethod
        def build(cls, obj):
            if isinstance(obj, abc.Mapping):
                return cls(obj)  # return a FrozenJson
            elif isinstance(obj, abc.MutableSequence):
                return [cls.build(item) for item in obj]  # build a list
            else:
                return obj
    # using __new__, not the classmethod build
    class FrozenJson:
        def __new__(cls, arg):  # actual constructor
            if isinstance(arg, abc.Mapping):
                return super().__new__(cls)  # delegate the constructor to super class
            elif isinstance(arg, abc.MutableSequence):
                return [cls(item) for item in arg]
            else:
                return arg
        def __init__(self, mapping):  # only an initializer
            self.__data = {}
            for key, value in mapping.items():
                if keyword.iskeyword(key):
                    key += '_'
                self.__data[key] = value
        def __getattr__(self, name):
            if hasattr(self.__data, name):
                return getattr(self.__data, name)
            else:
                return FrozenJson(self.__data[name])  # not calling .build(), but just call the constructor

    # restructure with shelve
    DB_NAME = os.path.join(os.path.dirname(__file__), 'schedule1_db')
    CONFERENCE = 'conference.115'
    class Record:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)  # attributed instances created from keyword arguments
    def load_db(db):
        raw_data = load()
        warnings.warn('loading' + DB_NAME)
        for collection, rec_list in raw_data['Schedule'].items():
            record_type = colletion[:-1]  # set with no trailling 's'
            for record in rec_list:
                key = '{}.{}'.format(record_type, record['serial'])
                record['serial'] = key  # update with the full key
                db[key] = Record(**record)  # build Record instance and save under the database
    import shelve
    db = shelve.open(DB_NAME)  # open or create a database
    if CONFERENCE not in db:  # check whether the database is populated
        load_db(db)  # load it
    db.close()  # always remember to close shelve.Shelve

    # retrieve record with properties
    import warnings
    import inspect
    DB_NAME = os.path.join(os.path.dirname(__file__), 'schedule2_db')
    CONFERENCE = 'conference.115'
    class Record:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __eq__(self, other):  # facilitate testing
            if isinstance(other, Record):
                return self.__dict__ == other.__dict__
            else:
                return NotImplemented
    class MissingDatabaseError(RuntimeError):
        """Raise when a database is required but was not set"""  # custom exceptions are marker class
    class DbRecord(Record):
        __db = None  # hold reference to opened database
        @staticmethod  # effect always the same, no matter how invoked
        def set_db(db):
            DbRecord.__db = db  # always get change in DbRecord class, not subclass
        @staticmethod
        def get_db():
            return DbRecord.__db
        @classmethod  # customize in subclass
        def fetch(cls, ident):
            db = cls.get_db()
            try:
                return db[ident]
            except TypeError:
                if db is None:
                    msg = "database not set; call '{}.set_db(my_db)'"
                    raise MissingDatabaseError(msg.format(cls.__name))
                else:
                    Raise  # reraise the exception as we don't know how to handle it.
        def __repr__(self):
            if hasattr(self, 'serial'):
                cls_name = self.__class__.__name__
                return '<{} serial={!r}>'.format(cls_name, self.serial)
            else:
                return super().__repr__()
    class Event(DbRecord):
        @property
        def venue(self):
            key = 'venue.{}'.format(self.venue_serial)
            return self.__class__.fetch(key)
        @property
        def speakers(self):
            if not hasattr(self, '_speaker_objs'):
                spkr_serials = self.__dict__['speakers']  # directly retrieve from instance __dict__
                fetch = self.__class__.fetch  # reference to the classmethod fetch(), incase there is a key named 'fetch'
                self._speaker_objs = [fetch('speaker.{}'.format(key)) for key in spkr_serials]
            return self._speaker_objs  # return the list built from fetching
        def __repr__(self):
            if hasattr(self, 'name'):
                cls_name = self.__class__.__name__
                return '<{} {!r}>'.format(cls_name, self.name)
            else:
                return super().__repr__()
    def load_db(db):
        raw_data = load()
        warnings.warn('loading' + DB_NAME)
        for collection, rec_list in raw_data['Schedule'].items():
            record_type = collection[:-1]
            cls_name = record_type.capitalize()  # to get a potential class name
            cls = globals().get(cls_name, DbRecord)  # get DbRecord if there is no such class name
            if inspect.isclass(cls) and issubclass(cls, DbRecord):
                factory = cls
            else:
                factory = DbRecord
            for record in rec_list:
                key = '{}.{}'.format(record_type, record['serial'])
                record['serial'] = key
                db[key] = factory(**record)  # construct the stored object with factory
    # LineItem with property
    class LineItem:
        def __init__(self, description, weight, price):
            self.description = description
            self.weight = weight
            self.price = price
        def subtotal(self):
            return self.weight * self.price
        @property  # a getter method
        def weight(self):
            return self.__weight  # value stored in a private value
        @preperty.setter  # bind two method together. protect from a negative value
        def weight(self, value):
            if value > 0:
                self.__weight = value[[[[yyt]]]]
            else:
                raise ValueError('value must be > 0')
    # property factory
    def quantity(storage_name):
        def qty_getter(instance):  # 'cls' will be strange for an instance, but can work
            return instance.__dict__[storage_name]  # bypass the property (by __dict__) to avoid recursive call.
        def qty_setter(instance, value):
            if value > 0:
                instance.__dict__[storage_name] = value
            else:
                raise ValueError('value must be > 0')
        return property(qty_getter, qty_setter)  # return a custom property object


if __name__ == "__main__":
    main()