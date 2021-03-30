import time


class Timing:
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.perf_counter()
            
    def __exit__(self, *args):
        self.end = time.perf_counter()
    
    @property
    def elapsed(self):
        return self.end - self.start

