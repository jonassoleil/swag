
class BaseModelIterator:
    def __init__(self):
        self.length = 0
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.i = -1

    def __len__(self):
        return self.length

    def get_next_model(self):
        raise NotImplementedError

    def __next__(self):
        self.i += 1
        if self.i >= self.length:
            raise StopIteration
        return self.get_next_model()
