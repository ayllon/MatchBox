class NoopListener(object):
    def __init__(self, iterable):
        self.__iterable = iterable

    def __iter__(self):
        return iter(self.__iterable)
