# tests/test_basic.py
import math

def helper_square(x: int) -> int:
    return x * x

def test_math_and_helper():
    # Deterministic, fast checks to verify the test runner works
    assert 2 + 2 == 4
    assert math.isclose(math.sqrt(9), 3.0)
    assert helper_square(5) == 25
