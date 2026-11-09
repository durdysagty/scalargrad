from scalargrad.value import Value
from abc import ABC, abstractmethod
from typing import Callable, Sequence
import random

class Module(ABC):

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

    @abstractmethod
    def parameters(self) -> list[Value]:
        ...

class Neuron(Module):

    def __init__(self, n_inputs: int, activation: Callable[[Value], Value] | None = Value.relu):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(0)
        self.activation = activation

    def __call__(self, inputs: Sequence[Value | float]) -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")
        weighted_sum = sum((wi * xi for wi, xi in zip(self.weights, inputs)), self.bias)
        return self.activation(weighted_sum) if self.activation else weighted_sum

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]

class Layer(Module):

    def __init__(self, n_inputs: int, n_neurons: int, activation: Callable[[Value], Value] | None = Value.relu):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def __call__(self, inputs: Sequence[Value | float]) -> Sequence[Value]:
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class MLP(Module):

    def __init__(self, n_inputs: int, l_sizes: list[int], activation: Callable[[Value], Value] | None = Value.relu, output_activation: Callable[[Value], Value] | None = None):
        if n_inputs <= 0:
            raise ValueError(f"n_inputs must be positive, got {n_inputs}")
        if not l_sizes:
            raise ValueError("l_sizes must have at least one layer size")
        if any(size <= 0 for size in l_sizes):
            raise ValueError(f"All layer sizes must be positive, got {l_sizes}")
        
        layer_sizes: list[int] = [n_inputs] + l_sizes
        n_layers = len(layer_sizes) - 1
        last_index = n_layers - 1
        self.layers: list[Layer] = [Layer(layer_sizes[i], layer_sizes[i + 1], activation=output_activation if i == last_index else activation) for i in range(n_layers)]

    def __call__(self, inputs: Sequence[Value | float]) -> Sequence[Value]:
        out = self.layers[0](inputs)
        for layer in self.layers[1:]:
            out = layer(out)
        return out

    def parameters(self) -> list[Value]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params