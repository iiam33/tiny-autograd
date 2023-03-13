class Number:
    def __init__(self, data, _children=(), grad=0.0, label=''):
        self.data = data
        self._prev = set(_children)
        self.grad = grad
        self.label = label

    def __repr__(self):
        return f'{self.data}'

    def __add__(self, other):
        return Number(self.data + other.data, (self, other))

    def __mul__(self, other):
        return Number(self.data * other.data, (self, other))
