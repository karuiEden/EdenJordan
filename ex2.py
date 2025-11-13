import sympy as sp

def cell_size(A, lam, k):
    B = (A - lam * sp.eye(A.shape[0]))
    return (B ** (k - 1)).rank() + (B ** (k + 1)).rank() - 2 * (B ** k).rank()

def jordan_form(A):
    vals = A.eigenvals()
    print(f'Алгебраические кратности собственных чисел {list(vals.keys())} равна {list(vals.values())}')
    rhos = []
    for val in vals.items():
        rho = len((A - val[0] * sp.eye(A.shape[0])).nullspace())
        print(f'Геометрическая кратность собственного числа {val[1]} равна {rho}\n')
        rhos.append(rho)
        for i in range(1, val[1] + 1):
            quantity = cell_size(A, val[0], i)
            if quantity > 0:
                print(f'Количество клеток числа {val[0]} размером {i} равно {quantity}')
        print()
    print(f'Исходя из суммарной геометрической кратности, количество Жордановых клеток равно {sum(rhos)}')
    print('Жордановы клетки:')
    sp.pprint(A.jordan_cells()[1])
    print('Жорданова форма:')
    sp.pprint(A.jordan_form()[1])

if __name__ == '__main__':
    #Ввести матрицу
    A = sp.Matrix(...)
    jordan_form(A)