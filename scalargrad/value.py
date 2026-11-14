from __future__ import annotations
import math

class Value:

    def __init__(self, data: float, _parents: tuple[Value, ...] = (), _op: str = '', label: str = ''):
        self.data = data
        self.grad = 0.0
        self._parents = set(_parents)
        self._op = _op
        self.label = label

        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}) {f', op={self._op}' if self._op else ''}"

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | float) -> Value:
        return self + other


    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __pow__(self, other: Value | float) -> Value:
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | float) -> Value:
        return other + (-self)

    def __neg__(self) -> Value:
        return self * -1

    def __truediv__(self, other: Value | float) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: Value | float) -> Value:
        return other * self**-1

    def tanh(self) -> Value:
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Compute d(self)/d(node) for every node in this value's graph.

        Gradients accumulate into .grad; they are not reset between calls.
        Set them to zero first (e.g. Module.zero_grad()) when you need a
        fresh pass.
        """

        topo: list[Value] = []
        visited: set[Value] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:
            v, parents_done = stack.pop()
            if parents_done:
                topo.append(v)                 # all its parents are already in topo
            elif v not in visited:
                visited.add(v)
                stack.append((v, True))        # revisit v after its parents
                for p in v._parents:
                    if p not in visited:
                        stack.append((p, False))

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()