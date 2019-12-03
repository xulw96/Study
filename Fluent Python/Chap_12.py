def main():
    # Multiple inheritance, MRO
    class A:
        def ping(self):
            print('ping:', self)
    class B(A):
        def pong(self):
            print('pong:', self)
    class C(A):
        def pong(self):
            print('pong:', self)
    class D(B, C):  # the order in subclassing change the order for MRO: method resolution order
        def ping(self):
            super().ping()
            print('post-ping:', self)
        def pingpong(self):
            self.ping()
            super().ping()  # find the method at class A
            self.pong()
            super().pong()
            C.pong(self)  # igonre the MRO and implement C.pong directly. (must input self)
    print(D().pingpong())
    def print_mro(cls):
        print(', '.join(c.__name__ for c in cls.__mro__))  # show the superclass of current class



if __name__ == "__main__":
    main()