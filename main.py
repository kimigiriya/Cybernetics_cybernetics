import numpy as np
import matplotlib.pyplot as plt
from sympy import parse_expr, symbols, fraction, simplify, N, arg, solve, Poly

nu = 10  # Смещение
omega = 2  # Частоты
Δt = 0.25  # Шаг дискретизации
phi = 1  # Начальные фазы
k = 100  # Кол-во точек
C = 1  # Амплитуда
err = 10**(-8)  # Машинный нуль

t = np.linspace(0, k, 1000)
#x = nu + C * np.sin(omega * t + phi)
x = 20 + 2*np.sin(5*t + 0.2) + 8 * np.sin(2*t + 1)

t_dense = 0
x_dense = []
z_dense = []
ttt = [0]

for i in range(k):
    #x_dense.append(nu + np.sin(omega*t_dense + phi))
    x_dense.append(20 + 2*np.sin(5*t_dense + 0.2) + 8 * np.sin(2*t_dense + 1))
    z_dense.append(np.arctan(omega * t_dense))
    t_dense += Δt
    ttt.append(t_dense)


print(f"Массив x_dense: {x_dense}")
print(f"Массив z_dense: {z_dense}")


def remove_zero_rows(matrix, err=1e-18):
    first_zero_row = None
    for i in range(matrix.shape[0]):
        if np.all(np.abs(matrix[i]) < err):
            first_zero_row = i
            break

    if first_zero_row is not None:
        return matrix[:first_zero_row]
    else:
        return matrix

def matrix_identifier(k, x_dense, err):
    matrix = np.zeros((k + 1, k))
    for i in range(k):
        if i == 0:
            matrix[0][0] = 1
            matrix[1][0] = x_dense[0]
        else:
            matrix[0][i] = 0
            matrix[1][i] = x_dense[i]

    l = k
    for i in range(2, k + 1):
        l -= 1
        if i == k - 1:
            raise Exception("Взято слишком маленькое k!")
        elif abs(matrix[i - 1][0]) >= err and abs(matrix[i - 1][1]) >= err:
            for j in range(l):
                matrix[i][j] = (matrix[i - 2][j + 1] / matrix[i - 2][0]) - (matrix[i - 1][j + 1] / matrix[i - 1][0])
        else:
            return remove_zero_rows(matrix)

try:
    matrix = matrix_identifier(k, x_dense, err)
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    print(f"Матрица: {matrix}")

    m = matrix.shape[0]
    n = int((m - 4) / 4)
    print(f"Число гармоник: {n}")

    z = symbols('z')
    str_drob = f"{matrix[1][0]}"
    for i in range(2, len(matrix)):
        str_drob += f"/(1 + {matrix[i][0]}*z**(-1)"
    str_drob += (')'*(len(matrix) - 2))
    print("Несвернутая дробь: ")
    print(str_drob)
    print()
    expr = parse_expr(str_drob, transformations='all')
    numenator, denumenator = fraction(simplify(expr))
    coef_num, cof_denum = numenator.as_poly(z).LC(), denumenator.as_poly(z).LC()
    numenator, denumenator = N(numenator / coef_num), N(denumenator / cof_denum)
    print("Числитель: ", numenator)
    print("Знаменатель: ", denumenator)
    print()
    print("Свернутая дробь: ")
    print(f"G(z) = ({numenator}) / ({denumenator})")
    print()
    solutions = Poly(denumenator, z).nroots()
    print("Решения: ", solutions)
    real_roots, complex_roots = [], []
    real_roots = [str(root) for root in solutions if root.is_real]
    complex_roots = [str(root) for root in solutions if not root.is_real]
    print("Действительные корни: ", real_roots)
    print("Комплексные корни: ", complex_roots)
    print("=========================")
    delta_t = 0.25
    w = [float(arg(root).evalf() / delta_t) for root in complex_roots[1::2]]
    print("Круговые частоты: ", w)
    M = []
    extternal_size = 1 + 2 * n
    for i in range(extternal_size):
        M_i = [1, ]
        for w_j in w:
            M_i.append(np.sin(float(ttt[i]*w_j)))
            M_i.append(np.cos(float(ttt[i]*w_j)))
        M.append(M_i)
    #print(M)
    M = np.array(M)
    #print(M.shape)
    S = np.linalg.solve(M, matrix[1][:extternal_size])
    #print(S)
    nu, A, B = S[0], S[1:len(S):2], S[2:len(S):2]
    C = [np.sqrt(k**2 + l**2) for k, l in zip(A, B)]
    print("Амплитуды: ", C)
    fi = [np.arctan(l / k) for k, l in zip(A, B)]
    print("Фазы: ", fi)
    print("Смещение: ", nu)

except Exception as e:
    print(f"Ошибка: {e}")


plt.plot(t, x, label='Исходная функция')
plt.xlabel('kΔt')
plt.ylabel('x(kΔt)')
plt.title('Метод идентификации')
plt.grid(True)
plt.legend()
plt.show()


ttt = np.array(ttt)
fig, axes = plt.subplots(int(n), 1, figsize=(10, 3*int(n)))
for i in range(int(n)):
    tt = np.arange(0, max(ttt), 0.001)
    yyy = C[i] * np.sin(w[i] * tt + fi[i])
    axes[i].plot(tt, yyy, linewidth=2)
    axes[i].set_xlabel('Время, t')
    axes[i].set_ylabel('Амплитуда')
    axes[i].set_title(f'Гармоника {i+1}: A={C[i]:.2f}, ω={w[i]:.2f}, φ={fi[i]:.2f}')
    axes[i].grid(True)
plt.tight_layout()
plt.show()

