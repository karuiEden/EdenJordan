import sympy as sp
from jordan_solver import Solver as JordanSolver




if __name__ == "__main__":
    A = sp.Matrix([[0, 1, 0], [-4, 4, 0], [-2, 1, 2]])
    solver = JordanSolver(A)
    sp.pprint(solver.build_jordan_basis())
    sp.pprint(A.jordan_form()[0])
