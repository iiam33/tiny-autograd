import math


class Number:
    def __init__(self, data, _children=(), label=''):
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self.grad = 0.0
        self.label = label

    def __repr__(self):
        return f'{self.data}'

    def __add__(self, other):
        other  = other if isinstance(other, Number) else Number(data=other)
        out = Number(data=self.data + other.data, _children=(self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other  = other if isinstance(other, Number) else Number(data=other)
        out = Number(data=self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)

        out = Number(data=t, _children=(self,))

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(root):
            if root not in visited:
                visited.add(root)

                for node in root._prev:
                    build_topo(node)

                topo.append(root)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
