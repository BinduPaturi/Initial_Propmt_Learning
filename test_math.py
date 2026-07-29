import pytest
from math_util import add, sub


@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (10, 20, 30)
])
def test_add_parameterize(a, b, expected):
    assert add(a, b) == expected


@pytest.mark.math
def test_add_fixture(numbers):
    a, b = numbers
    assert add(a, b) == 3


@pytest.mark.math
def test_sub(numbers):
    a, b = numbers
    assert sub(a, b) == 1


@pytest.mark.sanity
def test_add_zeros():

    assert add(0, 0) == 0
