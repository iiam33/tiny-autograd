from number import Number
import random


class Neuron:
    def __init__(self, no_of_input):
        self.w = [Number(random.uniform(-1, 1)) for _ in range(no_of_input)]
        self.b = Number(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()

        return out


class Layer:
    def __init__(self, no_of_input, no_of_output):
        self.neurons = [Neuron(no_of_input) for _ in range(no_of_output)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs


class MLP:
    def __init__(self, no_of_input, no_of_outputs):
        size = [no_of_input] + no_of_outputs
        self.layers = [Layer(size[i], size[i+1])
                       for i in range(len(no_of_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

