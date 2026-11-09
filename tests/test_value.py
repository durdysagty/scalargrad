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