from basis import Basis
from matrix import Matrix, inverse
from vector import Vector, VectorType
from subspace import Subspace
from typing import Optional, Union


class Transformation:
    """
    A linear transformation.

    Attributes:
        matrix (Matrix): A matrix representing the transformation
    """
    def __init__(self, matrix: Optional[Matrix] = None, **kwargs):
        """
        Parameters:
            matrix (Optional[Matrix]): A matrix representing the transformation
            **kwargs:
                points (Dict[VectorType, VectorType]): A map of pairs of input and output points of the transformation
        """
        if matrix is not None:
            self.matrix = Matrix(matrix) if type(matrix) is not Matrix else matrix
        elif "points" in kwargs:
            inputs = [Vector(v) if type(v) is not Vector else v for v in kwargs["points"].keys()]
            outputs = [Vector(v) if type(v) is not Vector else v for v in kwargs["points"].values()]
            self.matrix = Matrix(cols=outputs) @ inverse(Basis(*inputs).matrix)

    def apply(self, v: Union[VectorType, Subspace]):
        """Returns the result of applying the transformation to a vector or a subspace"""
        if type(v) is Vector:
            return self.matrix @ v
        if type(v) is list:
            return self.matrix @ Vector(v)
        if type(v) is Subspace:
            return Subspace(*[self.matrix @ g for g in v.generators])
