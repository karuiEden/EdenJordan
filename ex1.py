import sympy as sp
from jordan_solver import Solver as JordanSolver




if __name__ == "__main__":
    A = sp.Matrix(...)
    solver = JordanSolver(A)
    solver.jordan_form()
    solver.build_jordan_chains()
    solver.print_jordan_ladders()
    print('ПРОВЕРКА')
    print(f'{solver.P_manual.inv() * A * solver.P_manual == solver.P.inv() * A * solver.P}')