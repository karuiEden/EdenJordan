import sympy as sp
from jordan_solver import Solver as JordanSolver

if __name__ == '__main__':
    #Ввести матрицу
    #Пример ввода A = sp.Matrix([[0, 1, 0], [-4, 4, 0], [-2, 1, 2]])
    A = sp.Matrix(...)
    solver = JordanSolver(A)
    solver.jordan_form()