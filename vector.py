from math import acos
from copy import copy


class Vector:
    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"({', '.join(str(x) for x in self.elements)})"

    def __getitem__(self, index):
        return self.elements[index]

    def __setitem__(self, index, value):
        self.elements[index] = value

    def __add__(self, other):
        return Vector([x + y for x, y in zip(self.elements, other.elements)])

    def __sub__(self, other):
        return Vector([x - y for x, y in zip(self.elements, other.elements)])

    def __mul__(self, other):
        return Vector([x * other for x in self.elements])

    def __div__(self, other):
        return Vector([x * other for x in self.elements])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        return all([x == y for x, y in zip(self.elements, other.elements)])

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == len(self):
            raise StopIteration
        else:
            self.index += 1
            return self.elements[self.index - 1]

    def __copy__(self):
        return Vector(copy(self.elements))

    @property
    def dim(self):
        return 0

    @property
    def size(self):
        return (len(self),)

    def norm(self):
        return sum(x**2 for x in self.elements) ** .5

    def dot(self, other):
        if type(other) is list:
            other = Vector(other)
        return sum(x * y for x, y in zip(self.elements, other.elements))

    def cross(self, other):
        if type(other) is list:
            other = Vector(other)
        assert(self.dim == other.dim == 3)
        return Vector([self.elements[1] * other.elements[2] - self.elements[2] * other.elements[1],
                       self.elements[2] * other.elements[0] - self.elements[0] * other.elements[2],
                       self.elements[0] * other.elements[1] - self.elements[1] * other.elements[0]])

    def is_parallel(self, other):
        return self / self.norm() == other / other.norm()

    def is_perpendicular(self, other):
        return self.dot(other) == 0

    def angle(self, other):
        assert(self.is_parallel(other))
        return acos((self.dot(other)) / (self.norm() * other.norm()))
