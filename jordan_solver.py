from typing import Any

import sympy as sp


class Solver:

    def __init__(self, A: sp.Matrix):
        if A is not isinstance(A, sp.Matrix):
            self.matrix = sp.Matrix(A)
        else:
            self.matrix = A
        self.n = A.shape[0]
        self.I = sp.eye(self.n)
        self.J = None
        self.P = None
        self.step_counter = 0
        self.eigenvals = {}
        self.roots = {}
        self.jordan_cells_info = {}

    def _print_step(self, title: str, content: str = ""):
        self.step_counter += 1
        print(70 * '-')
        print(f'Шаг {self.step_counter}: {title}')
        print(70 * '-')
        if content:
            print(content)

    def cell_quantity(self, lam: int, k: int) -> int:
        B = (self.matrix - lam * sp.eye(self.n))
        ranks = tuple(((B ** i).rank() for i in range(k - 1, k + 2)))
        return ranks[0] + ranks[2] - 2 * ranks[1]

    def jordan_form(self) -> None:
        self._print_step("Собственные числа и их кратности")
        self.eigenvals = self.matrix.eigenvals()
        rhos = []
        print("\nАлгебраические кратности:")
        for val, mult in self.eigenvals.items():
            print(f'   λ = {val}, a = {mult}')
        print("\nГеометрические кратности:")
        for val, mult in self.eigenvals.items():
            rho = len((self.matrix - val * self.I).nullspace())
            print(f'   λ = {val}, ρ = {rho}')
            rhos.append(rho)
        self._print_step("Жордановы клетки")
        for val, mult in self.eigenvals.items():
            for i in range(1, mult + 1):
                quantity = self.cell_quantity(val, i)
                if quantity > 0:
                    print(f'  λ = {val}; Количество клеток размера {i}x{i}: {quantity} ')
        print(f'\nКоличество клеток Жордана равна сумме геометрических кратностей всех чисел. Оно равно {sum(rhos)}')
        print('Жордановы клетки:')
        self.jordan_cells_info = self.matrix.jordan_cells()[1]
        sp.pprint(self.jordan_cells_info)
        self._print_step("Жорданова форма")
        print("Строится путем построения блоков в диагональ, начинаем с клеток большего размера")
        self.J = self.matrix.jordan_form()[1]
        print("\nЖорданова форма:")
        sp.pprint(self.J)

    def find_root_vectors(self) -> dict[int, list[sp.Matrix]]:
        self._print_step("Поиск корневых векторов")
        if not self.eigenvals:
            self.eigenvals = self.matrix.eigenvals()
        for val, mult in self.eigenvals.items():
            print(f"\n  λ = {val}; a = {mult}")
            B = (self.matrix - val * self.I)
            all_roots = []
            for k in range(1, mult + 1):
                B_k = B ** k
                print(f"\nМатрица B = (A - λI) в степени {k}")
                sp.pprint(B_k)
                nullspace_k = B_k.nullspace()
                print(f'\n Степень k = {k}:')
                print(f'  rank(B^{k}) = {B_k.rank()}')
                print(f'  dim(ker(B^{k})) = {len(nullspace_k)}')

                if k == 1:
                    new_vectors = nullspace_k
                    if new_vectors:
                        print("  Собственные векторы:")
                        for i, vec in enumerate(new_vectors, 1):
                            print(f'    v_{i} = {vec}')
                else:
                    prev_nullspace = (B ** (k - 1)).nullspace()
                    new_vectors = [v for v in nullspace_k if v not in prev_nullspace]
                    if new_vectors:
                        print(f'    Корневые векторы порядка {k}:')
                        for i, vec in enumerate(new_vectors, 1):
                            print(f'    v_{i} = {vec}')

                all_roots.extend(new_vectors)
            self.roots[val] = all_roots
        return self.roots
