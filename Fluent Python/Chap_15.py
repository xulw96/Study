def main():
    # context manager
    class LookingGlass:
        def __enter__(self):
            import sys
            self.original_write = sys.stdout.write  # hold the method for later use
            sys.stdout.write = self.reverse_write  # output in reversed order, monkey patching
            return 'JABBERWOCKY'
        def reverse_write(self, text):
            self.original_write(text[::-1])
        def __exit__(self, exc_type, exc_val, exc_tb):
            import sys  # it's cheap because python cache it
            sys.stdout.write = self.original_write  # restore the method
            if exc_type is ZeroDivisionError:  # handle the exception
                print('Please DO NOT divide by zero')
                return True  # interpreter then suppress the exception
    # the @contextmanager
    import contextlib
    @contextlib.contextmanager  # wrap a generator function into a class with '__enter__' and '__exit__'
    def looking_glass():
        import sys
        original_write = sys.stdout.write
        def reverse_write(text):  # a custom reverse writing
            original_write(text[::-1])
        sys.stdout.write = reverse_write  # replace the writing method
        msg = ''
        try:  # handle the error incase the function abort without restoring to the original state
            yield 'JABBERWOCKY'  # this value will be bound to the 'as' clause
        except ZeroDivisionError:
            msg = 'Please DO NOT divide by zero!'
        finally:
            sys.stdout.write = original_write  # this will be executed after the control block
            if msg:
                print(msg)





if __name__ == "__main__":
    main()