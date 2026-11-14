from scalargrad.value import Value
import pytest

def test_add_backward():
    a, b = Value(2.0), Value(1.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0 and b.grad == 1.0

def test_mul_backward():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == 3.0 and b.grad == 2.0

@pytest.mark.parametrize("op", [
    lambda a, b: a + b,
    lambda a, b: a * b,
    lambda a, b: a - b,
    lambda a, b: a / b,
])
def test_ops_with_scalar_on_both_sides(op):
    a= Value(2.0)
    assert isinstance(op(a, 5.0), Value) and isinstance(op(5.0, a), Value)

@pytest.mark.parametrize("x", [Value(-1000.0), Value(-50.0), Value(0.0), Value(50.0), Value(1000.0)])
def test_tanh(x):
    y = x.tanh()
    assert -1 <= y.data <= 1, f"tanh({x.data}) = {y.data} is out of bounds"

def test_grad_sums_over_paths():
    a = Value(2.0)
    (a * a).backward()
    assert a.grad == 4.0          # d(a*a)/da = 2a, two paths of 2

def test_grad_accumulates_across_backward_calls():
    a = Value(2.0)
    b = a * a
    b.backward()
    b.backward()
    assert a.grad == 8.0          # not reset between calls
    a.grad = 0.0
    b.backward()
    assert a.grad == 4.0          # after resetting, one pass again


def test_grad_through_two_levels():
    a, b, c = Value(2.0), Value(3.0), Value(4.0)
    ((a * b) * c).backward()
    assert a.grad == 12.0
    assert b.grad == 8.0
    assert c.grad == 6.0

def test_grad_through_diamond():
    a, b = Value(2.0), Value(3.0)
    d = a * b
    c = a + 1
    (d * c).backward()
    assert a.grad == 15.0          # de/da = b*c + d = 9 + 6

def test_backward_on_leaf():
    leaf = Value(5.0)
    leaf.backward()
    assert leaf.grad == 1.0