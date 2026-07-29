from math_util import add


def test_greet_add(numbers):
    a, b = numbers
    assert add(a, b) == 3
