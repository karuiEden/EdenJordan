from typing import Any

import sympy as sp
from sympy import simplify, Matrix


class Solver:

    def __init__(self, A: sp.Matrix):
        if A is not isinstance(A, sp.Matrix):
            self.matrix = sp.Matrix(A)
        else:
            self.matrix = A
        self.n = A.shape[0]
        self.I = sp.eye(self.n)
        self.J = self.matrix.jordan_form()[1]
        self.P = self.matrix.jordan_form()[0]
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
        all_cells_info = []
        print("\nАлгебраические кратности:")
        for val, mult in self.eigenvals.items():
            print(f'   λ = {val}, a = {mult}')
        print("\nГеометрические кратности:")
        for val, mult in self.eigenvals.items():
            rho = len((self.matrix - val * self.I).nullspace())
            print(f'   λ = {val}, ρ = {rho}')
            rhos.append(rho)
            self._print_step("Жордановы клетки")
            cells_for_eigenval = []
            for i in range(1, mult + 1):
                quantity = self.cell_quantity(val, i)
                if quantity > 0:
                    print(f'  λ = {val}; Количество клеток размера {i}x{i}: {quantity} ')
                    for _ in range(quantity):
                        cells_for_eigenval.append(i)
                        all_cells_info.append({
                            'eigenvalue': val,
                            'size': i
                        })
            print(f"\n   Размеры клеток для λ = {val}: {cells_for_eigenval}")

        print(f'\nКоличество клеток Жордана равна сумме геометрических кратностей всех чисел. Оно равно {sum(rhos)}')
        print('Жордановы клетки:')
        sp.pprint(self.matrix.jordan_cells()[1])
        self.jordan_cells_info = all_cells_info
        self._print_step("Жорданова форма")
        print("Строится путем построения блоков в диагональ, начинаем с клеток большего размера")
        self.J = self.matrix.jordan_form()[1]
        print("\nЖорданова форма:")
        sp.pprint(self.J)


    def build_jordan_chains(self):

        self._print_step("Построение Жордановых цепочек из корневых векторов")

        if not self.jordan_cells_info:
            print("⚠ Сначала вызовите jordan_form()")
            return

        self.jordan_chains = []
        all_basis_vectors = []

        # Группируем клетки по собственным значениям
        cells_by_eigenvalue = {}
        for cell in self.jordan_cells_info:
            eigenval = cell['eigenvalue']
            size = cell['size']
            if eigenval not in cells_by_eigenvalue:
                cells_by_eigenvalue[eigenval] = []
            cells_by_eigenvalue[eigenval].append(size)

        # Для каждого собственного значения
        for eigenval, cell_sizes in cells_by_eigenvalue.items():
            print(f"\n{'─' * 60}")
            print(f"📌 Собственное значение λ = {eigenval}")
            print(f"   Размеры клеток: {cell_sizes}")

            A_shifted = self.matrix - eigenval * self.I

            # Сортируем размеры по убыванию
            cell_sizes_sorted = sorted(cell_sizes, reverse=True)
            used_vectors = []

            for idx, k in enumerate(cell_sizes_sorted, 1):
                print(f"\n   ═══ Клетка {idx} размера {k}×{k} ═══")

                # Находим корневой вектор порядка k
                v0 = self._find_root_vector(eigenval, k, used_vectors)

                if v0 is None:
                    print(f"   ⚠ Не найден корневой вектор порядка {k}")
                    continue

                print(f"\n   Корневой вектор v₀ порядка {k}:")
                print(f"   v₀ = {v0.T}")

                # Проверка
                check_k = simplify(A_shifted ** k * v0)
                check_k_minus_1 = simplify(A_shifted ** (k - 1) * v0) if k > 1 else v0

                print(f"\n   Проверка:")
                print(f"   (A - λI)^{k} v₀ = {check_k.T} {'✓ = 0' if check_k == sp.zeros(self.n, 1) else '✗ ≠ 0'}")
                if k > 1:
                    print(
                        f"   (A - λI)^{k - 1} v₀ = {check_k_minus_1.T} {'✓ ≠ 0' if check_k_minus_1 != sp.zeros(self.n, 1) else '✗ = 0'}")

                # Строим цепочку
                chain = []
                current = v0

                print(f"\n   Жорданова цепочка:")
                for i in range(k):
                    chain.append(current)
                    print(f"   v_{i} = {current.T}")
                    used_vectors.append(current)

                    if i < k - 1:
                        current = A_shifted * current
                        current = simplify(current)

                # Последний вектор должен быть собственным
                last_vec = chain[-1]
                check_eigen = simplify(A_shifted * last_vec)
                print(f"\n   Проверка последнего вектора (собственный):")
                print(
                    f"   (A - λI) v_{k - 1} = {check_eigen.T} {'✓ = 0' if check_eigen == sp.zeros(self.n, 1) else '✗ ≠ 0'}")

                self.jordan_chains.append({
                    'eigenvalue': eigenval,
                    'size': k,
                    'chain': chain
                })

                all_basis_vectors.extend(chain[::-1])

        # Формируем матрицу перехода из векторов
        if all_basis_vectors:
            self.P_manual = Matrix.hstack(*all_basis_vectors)

            print(f"\n{'═' * 70}")
            print(f"ПОСТРОЕННАЯ МАТРИЦА ПЕРЕХОДА P (вручную):")
            sp.pprint(self.P_manual)

        return self.jordan_chains

    def _find_root_vector(self, eigenval, k, used_vectors):
        B = self.matrix - eigenval * self.I

        B_k = B ** k
        nullspace_k = B_k.nullspace()

        if k > 1:
            B_k_minus_1 = B ** (k - 1)
            nullspace_k_minus_1 = B_k_minus_1.nullspace()
        else:
            nullspace_k_minus_1 = []

        for v in nullspace_k:
            # Проверяем, что v не в Ker(B^(k-1))
            if k > 1:
                if v in nullspace_k_minus_1:
                    continue

            if used_vectors:
                is_independent = True
                for used in used_vectors:
                    if simplify(v - used) == sp.zeros(self.n, 1):
                        is_independent = False
                        break

                if not is_independent:
                    continue

            return v

        return None

    def print_jordan_ladders(self):
        """
        Печатает Жордановы лестницы из уже найденных цепочек.
        """
        if not hasattr(self, 'jordan_chains') or not self.jordan_chains:
            print("⚠ Сначала вызовите build_jordan_chains()")
            return

        print("\n=== Жордановы лестницы ===\n")
        i = 1
        for idx, chain_info in enumerate(self.jordan_chains, 1):
            lam = chain_info['eigenvalue']
            size = chain_info['size']
            chain = chain_info['chain']
            chain = chain[::-1]
            print(f"Блок {idx}: λ = {lam}, размер = {size}x{size}")
            for v in chain:
                # Показываем вектор и стрелку, кроме последнего
                print(f"v_{i} = {v.T}", end="")
                if i <= len(chain) - 1:
                    print("  ─▶ ", end="")
                i += 1
            print("\n")  # Пустая строка между блоками