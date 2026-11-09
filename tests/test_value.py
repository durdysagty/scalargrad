from scalargrad.value import Value

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