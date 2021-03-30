from typing import Callable


class CallCounter(object):
    """
    Utility to keep track of how many times a given call is used
    """

    def __init__(self, test: Callable):
        self.__test = test
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        return self.__test(*args, **kwargs)
