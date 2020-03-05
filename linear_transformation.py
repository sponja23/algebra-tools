from basis import Basis
from matrix import Matrix, inverse
from vector import Vector
from subspace import Subspace


class Transformation:
    def __init__(self, matrix=None, **kwargs):
        if matrix is not None:
            self.matrix = Matrix(matrix) if type(matrix) is not Matrix else matrix
        elif "points" in kwargs:
            inputs = [Vector(v) if type(v) is not Vector else v for v in kwargs["points"].keys()]
            outputs = [Vector(v) if type(v) is not Vector else v for v in kwargs["points"].values()]
            self.matrix = Matrix(cols=outputs) @ inverse(Basis(*inputs).matrix)

    def apply(self, v):
        if type(v) is Vector:
            return self.matrix @ v
        if type(v) is Subspace:
            return Subspace(*[self.matrix @ g for g in v.generators])
