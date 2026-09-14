import numpy as np
import matplotlib.pyplot as plt


def run_ekf_simulation():
    # --- Инициализация параметров ---
    T = 0.2  # Шаг дискретизации (секунды)
    B = 2.0  # Колесная база робота
    W_R, W_L = 2.0, 2.0  # Радиусы правого и левого колес
    E, D = 0.0, 0.15  # Математическое ожидание и дисперсия шума

    U_R, U_L = 2.0, 4.0  # Управляющие воздействия на колеса

    # Списки для хранения истинной траектории
    x, y, r = [0.0], [0.0], [0.0]
    # Списки для хранения зашумленных наблюдений датчиков
    zx, zy, zr = [], [], []
    # Списки для хранения оценок фильтра EKF
    X, Y, Rr = [0.0], [0.0], [0.0]

    # Начальная ковариационная матрица ошибки
    P = np.array([[1.1, 0.0, 0.0],
                  [0.0, 1.1, 0.0],
                  [0.0, 0.0, 1.1]])

    # Матрица шума наблюдений
    R = np.array([[D, 0.0, 0.0],
                  [0.0, D, 0.0],
                  [0.0, 0.0, D]])

    # Матрица наблюдения (прямой замер всех трех координат)
    H = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

    # Расчет скоростей
    S_R = W_R * U_R
    S_L = W_L * U_L
    S_t = (S_R + S_L) / 2.0
    S_r = (S_R - S_L) / (2.0 * B)

    # --- Основной цикл симуляции ---
    for k in range(0, 31):
        # 1. Генерация истинной траектории (Ground Truth)
        r.append(r[k] + T * S_r)
        x.append(x[k] + T * S_t * np.cos(r[k]) - 0.5 * T * T * S_t * S_r * np.sin(r[k]))
        y.append(y[k] + T * S_t * np.sin(r[k]) + 0.5 * T * T * S_t * S_r * np.cos(r[k]))

        # 2. Моделирование зашумленных наблюдений датчиков
        w_x = np.random.normal(E, D)
        w_y = np.random.normal(E, D)
        w_r = np.random.normal(E, D)
        zx.append(x[-1] + w_x)
        zy.append(y[-1] + w_y)
        zr.append(r[-1] + w_r)

        # 3. Расширенный фильтр Калмана (EKF)
        # Этап прогноза (Prediction)
        Rr.append(Rr[k] + T * S_r)
        X.append(X[k] + T * S_t * np.cos(Rr[k]) - 0.5 * T * T * S_t * S_r * np.sin(Rr[k]))
        Y.append(Y[k] + T * S_t * np.sin(Rr[k]) + 0.5 * T * T * S_t * S_r * np.cos(Rr[k]))

        # Вычисление Якобиана матрицы перехода состояний (F)
        F = np.array([[1.0, 0.0, T * S_t * (-np.sin(Rr[-1])) - 0.5 * T * T * S_t * S_r * np.cos(Rr[-1])],
                      [0.0, 1.0, T * S_t * np.cos(Rr[-1]) - 0.5 * T * T * S_t * S_r * np.sin(Rr[-1])],
                      [0.0, 0.0, 1.0]]).T

        P = np.matmul(np.matmul(F, P), F.T)

        # Этап коррекции (Update)
        S = np.matmul(np.matmul(H, P), H.T) + R
        K = np.matmul(np.matmul(P, H.T), np.linalg.inv(S))

        z_k = np.array([zx[-1], zy[-1], zr[-1]]).T
        x_pred = np.array([X[-1], Y[-1], Rr[-1]]).T

        mk = x_pred + np.matmul(K, (z_k - x_pred))
        P = P - np.matmul(np.matmul(K, S), K.T)

        X[-1] = mk[0]
        Y[-1] = mk[1]
        Rr[-1] = mk[2]

    # --- Визуализация результатов ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Истинная траектория (Ground Truth)", marker='o', color='green')
    plt.plot(zx, zy, label="Зашумленные наблюдения", marker='.', linestyle='None', alpha=0.5, color='red')
    plt.plot(X, Y, label="Оценка EKF", marker='+', color='blue')

    plt.legend()
    plt.title('Локализация робота с помощью Расширенного Фильтра Калмана (EKF)')
    plt.xlabel('Координата X')
    plt.ylabel('Координата Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run_ekf_simulation()
