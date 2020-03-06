from math import acos
from copy import copy
from functools import wraps
from typing import Union, List, Tuple

Scalar = Union[int, float]


def vector_argument(f):
    @wraps(f)
    def fixed_argument(self, other):
        if type(other) is not Vector:
            return f(self, Vector(other))
        else:
            return f(self, other)
    return fixed_argument


class Vector:
    """
    A vector.

    Attributes:
        elements (List[Scalar]): The vector's elements
        dim (int): The dimension of the vector
    """
    def __init__(self, elements: List[Scalar]) -> None:
        """
        Parameters:
            elements (List[Scalar]): The vector's elements
        """
        self.elements = list(elements)

    def __repr__(self) -> str:
        """Returns a representation of the vector"""
        return f"({', '.join(str(x) for x in self.elements)})"

    def __getitem__(self, index: int) -> Scalar:
        """Returns the element at a given index"""
        return self.elements[index]

    def __setitem__(self, index: int, value: Scalar) -> None:
        """Sets the element at a given index"""
        self.elements[index] = value

    @vector_argument
    def __add__(self, other: "Vector") -> "Vector":
        """Returns the sum of two vectors"""
        return Vector([x + y for x, y in zip(self.elements, other.elements)])

    @vector_argument
    def __sub__(self, other: "Vector") -> "Vector":
        """Returns the substraction of two vectors"""
        return Vector([x - y for x, y in zip(self.elements, other.elements)])

    def __mul__(self, other: Scalar) -> "Vector":
        """Returns the product of a vector and a scalar"""
        return Vector([x * other for x in self.elements])

    def __truediv__(self, other: Scalar) -> "Vector":
        """Returns the division between a vector and a scalar"""
        return Vector([x / other for x in self.elements])

    def __rmul__(self, other: Scalar) -> "Vector":
        """See Vector.__mul__"""
        return self.__mul__(other)

    @vector_argument
    def __eq__(self, other: "Vector") -> bool:
        """Returns whether two vectors are equal"""
        return all(x == y for x, y in zip(self.elements, other.elements))

    @vector_argument
    def __ne__(self, other: "Vector") -> bool:
        """Returns whether two vectors are different"""
        return not self == other

    def __len__(self):
        """Returns the length of the vector"""
        return len(self.elements)

    def __iter__(self) -> "Vector":
        self.index = 0
        return self

    def __next__(self) -> Scalar:
        if self.index == len(self):
            raise StopIteration
        else:
            self.index += 1
            return self.elements[self.index - 1]

    def __copy__(self) -> None:
        return Vector(copy(self.elements))

    @property
    def dim(self) -> int:
        """See Vector.__len__"""
        return len(self)

    @property
    def size(self) -> Tuple[int]:
        """Returns a 1-tuple with the vector's size"""
        return (len(self),)

    def norm(self) -> Scalar:
        """Returns the vector's magnitude"""
        return sum(x**2 for x in self.elements) ** .5

    @vector_argument
    def dot(self, other: "Vector") -> Scalar:
        """Returns the dot product between two vectors"""
        return sum(x * y for x, y in zip(self.elements, other.elements))

    @vector_argument
    def cross(self, other: "Vector") -> "Vector":
        """Returns the cross products between two vectors"""
        assert(self.dim == other.dim == 3)
        return Vector([self.elements[1] * other.elements[2] - self.elements[2] * other.elements[1],
                       self.elements[2] * other.elements[0] - self.elements[0] * other.elements[2],
                       self.elements[0] * other.elements[1] - self.elements[1] * other.elements[0]])

    @vector_argument
    def is_parallel(self, other: "Vector") -> bool:
        """Returns whether two vectors are parallel"""
        return self / self.norm() == other / other.norm()

    @vector_argument
    def is_perpendicular(self, other: "Vector") -> bool:
        """Returns whether two vectors are perpendicular"""
        return self.dot(other) == 0

    @vector_argument
    def angle(self, other: "Vector") -> float:
        """Returns the angle between two vectors"""
        assert(self.is_parallel(other))
        return acos((self.dot(other)) / (self.norm() * other.norm()))


VectorType = Union[Vector, List[Scalar]]
