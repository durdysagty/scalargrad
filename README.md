# scalargrad

A tiny scalar-valued autograd engine and a small neural-network library built on top of it, written from scratch as a learning project.

It follows Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and his video [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0). Compared with micrograd it adds type hints, an abstract `Module` base class, input validation and tests.

## What's inside

- `scalargrad/value.py` — `Value`, a scalar that records the operations applied to it and computes gradients with `backward()`. Supports `+`, `-`, `*` and `/` (with plain numbers on either side), `**` with a plain number as the exponent, plus `tanh()` and `relu()`.
- `scalargrad/nn.py` — `Neuron`, `Layer` and `MLP`, built from `Value`s, with a `Module` base class that provides `parameters()` and `zero_grad()`.
- `demo.ipynb` — trains a small MLP on a toy dataset and checks the hidden layers for dead ReLU neurons.
- `tests/` — pytest tests for `Value`.

## Example

```python
from scalargrad.value import Value
from scalargrad.nn import MLP

a = Value(2.0)
b = Value(-3.0)
c = a * b + a ** 2
c.backward()
print(a.grad, b.grad)   # 1.0 2.0  (dc/da = b + 2a, dc/db = a)

n = MLP(3, [4, 4, 1], output_activation=Value.tanh)
out = n([2.0, 3.0, -1.0])[0]
print(len(n.parameters()))   # 41
```

## Setup

Requires Python 3.10 or newer. The library itself has no dependencies; `pytest` is only needed for the tests and `ipykernel` for the notebook.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest ipykernel
pytest
```
