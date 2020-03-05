from matrix import Matrix, Vector, inverse
from subspace import Subspace
from fd import fd


class Basis:
    def __init__(self, *vectors):
        assert(all([len(v) == len(vectors[0]) for v in vectors]))
        assert(Matrix(vectors).determinant() != 0)
        self.elements = [v if type(v) is Vector else Vector(v) for v in vectors]

    @property
    def matrix(self):
        return Matrix(cols=self.elements)

    def get_subspace(self):
        return Subspace(*self.elements)

    def to_coords(self, vector):
        assert(self.matrix.determinant() != 0)
        return inverse(self.matrix) @ vector

    def from_coords(self, vector):
        return self.matrix @ vector

    def __repr__(self):
        return '{' + ", ".join([str(b) for b in self.elements]) + '}'

    def cob_matrix(frm, to): # Change of basis matrix
        return Matrix(cols=[to.to_coords(v) for v in frm.elements])

    def canonical_basis(dim):
        return Basis(*[Vector([1 if j == i else 0 for j in range(dim)]) for i in range(dim)])


fd["Basis"] = Basis

E = Basis([1, 0, 0], [0, 1, 0], [0, 0, 1])
b1 = Basis([1, 0, 0], [1, 1, 0], [1, 1, 1])
b2 = Basis([1, 1, 1], [0, 1, 1], [0, 1, 0])
b3 = Basis([1, 2, 3], [3, 2, 1], [0, 1, 0])
