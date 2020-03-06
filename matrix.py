from vector import Vector, VectorType, Scalar
from fd import fd
from copy import copy, deepcopy
from typing import List, Union, Optional, Tuple


class Matrix:
    """
    A matrix.

    Attributes:
        elements (List[List[Scalar]]): Its rows
    """
    def __init__(self, elements: Optional[List[VectorType]] = None, **kwargs) -> None:
        """
        Parameters:
            elements (List[VectorType]): The matrix's rows
            **kwargs:
                cols (List[VectorType]): The matrix's columns
        """
        if elements is not None:
            self.elements = [e if type(e) is not Vector else e.elements for e in elements]
        elif "cols" in kwargs:
            self.elements = Matrix(kwargs["cols"]).cols()
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        """Returns a representation of the matrix"""
        return "[[" + '],\n ['.join(', '.join(str(x) for x in row) for row in self.rows()) + "]]"

    def __getitem__(self, index: Union[Tuple[int], int]) -> Union[Scalar, Vector]:
        """Returns the row or number in the given index"""
        if type(index) is tuple and len(index) == 2:
            if type(index[0]) is slice:
                return Matrix([row[index[1]] for row in self.elements[index[0]]])
            return self.elements[index[0]][index[1]]
        return Vector(self.elements[index])

    def __setitem__(self, index: Union[Tuple[int], int], value: Union[Scalar, VectorType]):
        """Sets the row or number in the given index"""
        if type(index) is tuple and len(index) == 2:
            self.elements[index[0]][index[1]] = value
        else:
            self.elements[index] = value if type(value) is not Vector else value.elements

    def __len__(self):
        """Returns the number of rows"""
        return len(self.elements)

    def row(self, index: int) -> Vector:
        """Returns the row in a given index"""
        return Vector(self.elements[index])

    def col(self, index: int) -> Vector:
        """Returns the column in a given index"""
        return Vector([row[index] for row in self.elements])

    def rows(self) -> List[Vector]:
        """Returns a list of rows"""
        return [self.row(i) for i in range(self.size[0])]

    def cols(self):
        """Returns a list of columns"""
        return [self.col(i) for i in range(self.size[1])]

    @property
    def dim(self):
        """Returns the dimension of the matrix"""
        return len(self.elements) * len(self.elements[0])

    @property
    def size(self):
        """Returns a 2-tuple with the width and height of the matrix"""
        return (len(self.elements), len(self.elements[0]))

    def __add__(self, other: "Matrix") -> "Matrix":
        """Returns the sum of two matrices"""
        return Matrix([self_row + other_row for self_row, other_row in zip(self.rows(), other.rows())])

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Returns the substraction between two matrices"""
        return Matrix([self_row - other_row for self_row, other_row in zip(self.rows(), other.rows())])

    def __mul__(self, other: Scalar) -> "Matrix":
        """Returns the product of a scalar and a matrix"""
        return Matrix([other * row for row in self.rows()])

    def __rmul__(self, other: Scalar) -> "Matrix":
        """See Matrix.__mul__"""
        return self.__mul__(other)

    def __eq__(self, other: "Matrix") -> "Matrix":
        """Returns whether two matrices are equal"""
        return all(all(x == y for x, y in zip(row_x, row_y)) for row_x, row_y in zip(self.elements, other.elements))

    def __ne__(self, other: "Matrix") -> "Matrix":
        """Returns whether two matrices are different"""
        return not self == other

    def __matmul__(self, other: Union[VectorType, "Matrix"]) -> Union[Vector, "Matrix"]:
        """Returns the matrix multiplication between a matrix and a vector or another matrix"""
        assert(self.size[1] == len(other))
        if type(other) is list:
            other = Vector(other)
        if type(other) is Vector:
            return Vector([row.dot(other) for row in self.rows()])
        elif type(other) is Matrix:
            return Matrix([[row.dot(col) for col in other.cols()] for row in self.rows()])

    def __copy__(self) -> "Matrix":
        return Matrix(copy(self.elements))

    def __deepcopy__(self, *args) -> "Matrix":
        return Matrix(deepcopy(self.elements, *args))

    def submatrix(self, removed_row: int, removed_column: int) -> "Matrix":
        """Returns the same matrix without a row and a column"""
        return Matrix([[n for j, n in enumerate(row) if j != removed_column] for i, row in enumerate(self.elements) if i != removed_row])

    def determinant(self) -> int:  # needs fix
        """Returns the matrix's determinant"""
        assert(self.size[0] == self.size[1])
        if self.size == (2, 2):
            return self.elements[0][0] * self.elements[1][1] - self.elements[0][1] * self.elements[1][0]
        result = 0
        for i, n in enumerate(self.elements[0]):
            result += ((i % 2) * -2 + 1) * n * self.submatrix(0, i).determinant()
        return result

    def transpose(self) -> "Matrix":
        """Returns the matrix's transpose"""
        return Matrix(self.cols())

    def apply_row_operation(self, operation):
        self = operation @ self

    def apply_col_operation(self, operation):
        self = self @ operation


def identity_matrix(size: Tuple[int, int]) -> Matrix:
    return Matrix([[1 if i == j else 0 for j in range(size[1])] for i in range(size[0])])


def swap_rows(matrix: Matrix, i: int, j: int) -> None:
    tmp = copy(matrix[i])
    matrix[i] = matrix[j]
    matrix[j] = tmp


def multiply_row(matrix: Matrix, i: int, n: Scalar) -> None:
    matrix[i] = matrix[i] * n


def combine_rows(matrix: Matrix, i: int, j: int, n: Scalar) -> None:
    matrix[i] = matrix[i] + n * matrix[j]


def elemental_matrix(size: Tuple[int, int], type: str, i: int, j: Scalar, k: Scalar = 0) -> Matrix:
    result = identity_matrix(size)
    if type[0] == 's':
        result[i, i] = 0
        result[i, j] = 1
        result[j, j] = 0
        result[j, i] = 1
    if type[0] == 'm':
        result[i, i] = j
    if type[0] == 'c':
        result[i, j] = k
    return result


def row_echelon_form(matrix: Matrix, return_transform: bool = False) -> Matrix:
    """Returns the matrix in row echelon form and, if return_transform is True, the transformation to get it there"""
    m = deepcopy(matrix)
    transform = identity_matrix((m.size[0], m.size[0]))
    for i in range(min(m.size[0], m.size[1])):
        if m[i, i] == 0:  # Find non-zero row
            for k in range(i + 1, m.size[0]):
                if m[k, i] != 0:
                    transform = elemental_matrix(transform.size, 'swap', i, k) @ transform
                    swap_rows(m, i, k)
                    break
            else:
                continue
        for j in range(i + 1, m.size[0]):
            transform = elemental_matrix(transform.size, 'combine', j, i, -m[j, i]/m[i, i]) @ transform
            combine_rows(m, j, i, -m[j, i]/m[i, i])
    return (m, transform) if return_transform else m


def inverse(matrix: Matrix) -> Matrix:
    """Returns the inverse of a square matrix"""
    assert(matrix.size[0] == matrix.size[1])
    m, result = row_echelon_form(matrix, True)
    for i in range(m.size[1]):
        if m[i, i] != 0:
            result = elemental_matrix(m.size, 'multiply', i, 1/m[i, i]) @ result
            multiply_row(m, i, 1/m[i, i])
            for j in range(0, i):
                result = elemental_matrix(m.size, 'combine', j, i, -m[j, i]) @ result
                combine_rows(m, j, i, -m[j, i])
    return result


def is_consistent(matrix: Matrix, vector: Vector) -> bool:
    """Returns whether the matrix and the vector form a consistent system of equations"""
    assert(len(vector) == matrix.size[0])
    if matrix.size[0] == matrix.size[1] and matrix.determinant() != 0:
        return True
    m, transform = row_echelon_form(matrix, True)
    v = transform @ vector
    for i in range(m.size[0]):
        if all([n == 0 for n in m[i]]) and v[i] != 0:
            return False
    return True


def is_echelon_form(matrix: Matrix) -> bool:
    """Returns whether a matrix is in echelon form"""
    if matrix[0, 0] == 0:
        return False
    prev_col_index = 0
    for row in matrix.rows()[1:]:
        for i, element in enumerate(row):
            if element != 0:
                if i <= prev_col_index:
                    return False
                else:
                    prev_col_index = i
                    break
    return True


def col_space(matrix):
    return fd["Subspace"](*matrix.elements)


def row_space(matrix):
    return col_space(matrix.transpose())


def null_space(matrix):
    return fd["Subspace"](equations=matrix)


m1 = Matrix([[1, 2, 2], [2, 3, 1], [1, 1, 1]])
m2 = Matrix([[5, 2, 1], [0, 2, 1], [0, 0, 3]])
m3 = Matrix([[1, 0, 0], [2, 0, 2], [0, 1, 0]])
m4 = Matrix([[1, 2, 3], [2, 1, 3], [3, 2, 1], [1, 1, 2]])
m5 = Matrix([[1, 2, 4, 0], [4, 4, 0, 6], [1, 1, 2, 4]])
m6 = Matrix([[1, 1], [0, 1], [1, 1]])
