from copy import copy
from vector import Vector
from fd import fd
from matrix import Matrix, is_consistent, row_echelon_form
from basis import Basis


class Subspace:
    def __init__(self, *vectors, **kwargs):
        assert(all([len(v) == len(vectors[0]) for v in vectors]))
        if len(vectors) > 0:
            self.generators = [v if type(v) is Vector else Vector(v) for v in vectors]
        elif "equations" in kwargs:
            self.generators = Subspace(*kwargs["equations"]).equations()  # Double orthogonal complement
        if kwargs.get("remove_extra", False):
            self.remove_extra_generators()

    @property
    def generator_matrix(self):
        return Matrix(cols=self.generators)

    def basis(self):
        result = copy(self.generators)
        for i, v in enumerate(result):
            if is_consistent(Matrix(cols=result[:i] + result[i + 1:]), v):
                result.pop(i)
        return Basis(result)

    def dim(self):
        return len(self.get_base())

    def contains(self, other):
        if type(other) is Subspace:
            return all([self.contains(v) for v in other.generators])
        return is_consistent(self.generator_matrix, other)

    def remove_extra_generators(self):
        self = self.get_base().get_subspace()

    def equations(self):
        m, transform = row_echelon_form(self.generator_matrix, True)
        return [transform[i] for i in range(m.size[0]) if all([n == 0 for n in m[i]])]

    def __add__(self, other):
        return Subspace(*(self.generators + other.generators))

    def intersection(self, other):
        return Subspace(equations=self.equations() + other.equations())

    def __eq__(self, other):
        return all([self.contains(v) for v in other.generators]) and all([other.contains(v) for v in self.generators])

    def __repr__(self):
        return '<' + ", ".join([str(b) for b in self.generators]) + '>'

    def orthogonal_complement(self):
        return Subspace(equations=self.generators)


fd["Subspace"] = Subspace

abc = "abcdefghijklmnopqrstuvwxyz"


class SolutionSpace:
    def __init__(self, origin, *generators):
        self.origin = origin if type(origin) is Vector else Vector(origin)
        self.subspace = Subspace(*generators)

    def contains(self, vector):
        if type(vector) is not Vector:
            vector = Vector(vector)
        return self.subspace.contains(vector - self.origin)

    def __repr__(self):
        return str(self.origin) + ' + ' + ' + '.join([abc[i] + str(g) for i, g in enumerate(self.subspace.generators)])


def str_sign(n):
    return '-' if n < 0 else '+'


def pp_equations(equations):
    result = ""
    for eq in equations:
        result += ' '.join([f"{str_sign(n)} {abs(n) if abs(n) != 1 else ''}x{i + 1}" for i, n in enumerate(eq) if n != 0]) + " = 0\n"
    result = result[:-1]
    print(result)


r3 = Subspace([1, 0, 0], [0, 1, 0], [0, 0, 1])
s1 = Subspace([1, 0, 0], [1, 1, 0])
s2 = Subspace([1, 0, 1], [1, 1, 1], [2, 1, 2])
s3 = Subspace([0, 1, 0], [1, 0, 0])
s4 = Subspace([1, 1, 1], [2, 0, 2])
s5 = Subspace(equations=[[1, 1, 0], [0, 0, 1]])

r2 = Subspace([1, 0], [0, 1])
s6 = Subspace([1, -1])
ss1 = SolutionSpace([1, 1], [-1, 1])
