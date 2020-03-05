from vector import Vector
from fd import fd
from copy import copy, deepcopy


class Matrix:
    def __init__(self, elements=None, **kwargs):
        if elements is not None:
            self.elements = [e if type(e) is not Vector else e.elements for e in elements]
        elif "cols" in kwargs:
            self.elements = Matrix(kwargs["cols"]).transpose().elements
        else:
            raise NotImplementedError

    def __repr__(self):
        return "[[" + '],\n ['.join(', '.join(str(x) for x in row) for row in self.rows()) + "]]"

    def __getitem__(self, index):
        if type(index) is tuple and len(index) == 2:
            if type(index[0]) is slice:
                return Matrix([row[index[1]] for row in self.elements[index[0]]])
            return self.elements[index[0]][index[1]]
        return Vector(self.elements[index])

    def __setitem__(self, index, value):
        if type(index) is tuple and len(index) == 2:
            self.elements[index[0]][index[1]] = value
        else:
            self.elements[index] = value if type(value) is not Vector else value.elements

    def __len__(self):
        return len(self.elements)

    def row(self, index):
        return Vector(self.elements[index])

    def col(self, index):
        return Vector([row[index] for row in self.elements])

    def rows(self):
        return [self.row(i) for i in range(self.size[0])]

    def cols(self):
        return [self.col(i) for i in range(self.size[1])]

    @property
    def dim(self):
        return len(self.elements) * len(self.elements[0])

    @property
    def size(self):
        return (len(self.elements), len(self.elements[0]))

    def __add__(self, other):
        return Matrix([self_row + other_row for self_row, other_row in zip(self.rows(), other.rows())])

    def __sub__(self, other):
        return Matrix([self_row - other_row for self_row, other_row in zip(self.rows(), other.rows())])

    def __mul__(self, other):
        return Matrix([other * row for row in self.rows()])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        return all([all([x == y for x, y in zip(row_x, row_y)]) for row_x, row_y in zip(self.elements, other.elements)])

    def __ne__(self, other):
        return not self == other

    def __matmul__(self, other):
        assert(self.size[1] == len(other))
        if type(other) is list:
            other = Vector(other)
        if type(other) is Vector:
            return Vector([row.dot(other) for row in self.rows()])
        elif type(other) is Matrix:
            return Matrix([[row.dot(col) for col in other.cols()] for row in self.rows()])

    def __copy__(self):
        return Matrix(copy(self.elements))

    def __deepcopy__(self, *args):
        return Matrix(deepcopy(self.elements))

    def submatrix(self, removed_row, removed_column):
        return Matrix([[n for j, n in enumerate(row) if j != removed_column] for i, row in enumerate(self.elements) if i != removed_row])

    def determinant(self):  # needs fix
        assert(self.size[0] == self.size[1])
        if self.size == (2, 2):
            return self.elements[0][0] * self.elements[1][1] - self.elements[0][1] * self.elements[1][0]
        result = 0
        for i, n in enumerate(self.elements[0]):
            result += ((i % 2) * -2 + 1) * n * self.submatrix(0, i).determinant()
        return result

    def transpose(self):
        return Matrix(self.cols())

    def apply_row_operation(self, operation):
        self = operation @ self

    def apply_col_operation(self, operation):
        self = self @ operation


def identity_matrix(size):
    return Matrix([[1 if i == j else 0 for j in range(size[1])] for i in range(size[0])])


def swap_rows(matrix, i, j):
    tmp = copy(matrix[i])
    matrix[i] = matrix[j]
    matrix[j] = tmp


def multiply_row(matrix, i, n):
    matrix[i] = matrix[i] * n


def combine_rows(matrix, i, j, n):
    matrix[i] = matrix[i] + n * matrix[j]


def elemental_matrix(size, type, i, j, k=0):
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


def row_echelon_form(matrix, return_transform=False):
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


def inverse(matrix):
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


def is_consistent(matrix, vector):
    assert(len(vector) == matrix.size[0])
    if matrix.size[0] == matrix.size[1] and matrix.determinant() != 0:
        return True
    m, transform = row_echelon_form(matrix, True)
    v = transform @ vector
    for i in range(m.size[0]):
        if all([n == 0 for n in m[i]]) and v[i] != 0:
            return False
    return True


def is_echelon_form(matrix):
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
