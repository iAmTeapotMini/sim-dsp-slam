import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_sensors(time, arr):
    """Фильтрация строк измерений для текущего шага времени."""
    return np.array([a for a in arr if a[0] == time])


def get_jacobi_matrix_h(sensor_coords, robot_coords):
    """Вычисление матрицы Якобиана H для модели наблюдения расстояния."""
    sensor_x, sensor_y = sensor_coords
    robot_coords_x, robot_coords_y = robot_coords
    dx = robot_coords_x - sensor_x
    dy = robot_coords_y - sensor_y
    distance = np.sqrt(dx ** 2 + dy ** 2)
    return np.array([dx / distance, dy / distance, 0.0])


def ekf_localization(deltas, sensors, landmarks):
    """Расширенный фильтр Калмана (EKF)"""
    noise_variance = 0.2

    P = np.array([[5.1, 0.0, 0.0],
                  [0.0, 5.1, 0.0],
                  [0.0, 0.0, 5.1]])

    Q = np.array([[noise_variance, 0.0, 0.0],
                  [0.0, noise_variance, 0.0],
                  [0.0, 0.0, noise_variance]])

    X, Y, ekf_theta = [0.0], [0.0], [0.0]

    for t, d in enumerate(deltas):
        delta_r1, delta_t, delta_r2 = d

        # --- Прогноз (Prediction) ---
        X.append(X[-1] + delta_t * np.cos(ekf_theta[-1] + delta_r1))
        Y.append(Y[-1] + delta_t * np.sin(ekf_theta[-1] + delta_r1))
        ekf_theta.append(ekf_theta[-1] + delta_r1 + delta_r2)

        F = np.array([[1.0, 0.0, -delta_t * np.sin(ekf_theta[-2] + delta_r1)],
                      [0.0, 1.0, delta_t * np.cos(ekf_theta[-2] + delta_r1)],
                      [0.0, 0.0, 1.0]]).T
        P = F @ P @ F.T + Q

        # --- Коррекция (Update) ---
        sens = get_sensors(t + 1, sensors)
        m = len(sens)
        if m == 0:
            continue

        R = np.diag(np.full(m, noise_variance))
        H = np.empty((m, 3))
        true_r = np.empty(m)
        pred_r = np.empty(m)

        for i, s in enumerate(sens):
            r_x, r_y = landmarks[int(s[1] - 1)]
            true_r[i] = s[2]
            pred_r[i] = np.sqrt((r_x - X[-1]) ** 2 + (r_y - Y[-1]) ** 2)
            H[i] = get_jacobi_matrix_h(landmarks[int(s[1] - 1)], (X[-1], Y[-1]))

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        mk = np.array([X[-1], Y[-1], ekf_theta[-1]]) + K @ (true_r - pred_r)
        P = P - K @ S @ K.T

        X[-1] = mk[0]
        Y[-1] = mk[1]
        ekf_theta[-1] = mk[2]

    return X, Y


def ukf_localization(deltas, sensors, landmarks):
    """Сигма-точечный фильтр Калмана (UKF)"""
    noise_variance = 0.2
    n = 3

    P = np.array([[5.1, 0.0, 0.0],
                  [0.0, 5.1, 0.0],
                  [0.0, 0.0, 5.1]])

    Q = np.array([[noise_variance, 0.0, 0.0],
                  [0.0, noise_variance, 0.0],
                  [0.0, 0.0, noise_variance]])

    X, Y, theta = [0.0], [0.0], [0.0]
    SP_x = [0.0 for _ in range(2 * n + 1)]
    SP_y = [0.0 for _ in range(2 * n + 1)]
    SP_theta = [0.0 for _ in range(2 * n + 1)]
    alpha, beta = 1, 2

    for t, d in enumerate(deltas):
        delta_r1, delta_t, delta_r2 = d
        lambda_ = alpha ** 2 * (n + t) - n

        sqrt_P = np.linalg.cholesky(P)

        SP_x[0] = X[t]
        SP_y[0] = Y[t]
        SP_theta[0] = theta[t]

        for i in range(1, n + 1):
            scale = np.sqrt(n + lambda_)
            SP_x[i] = X[t] + scale * sqrt_P[0][i - 1]
            SP_y[i] = Y[t] + scale * sqrt_P[1][i - 1]
            SP_theta[i] = theta[t] + scale * sqrt_P[2][i - 1]

            SP_x[i + n] = X[t] - scale * sqrt_P[0][i - 1]
            SP_y[i + n] = Y[t] - scale * sqrt_P[1][i - 1]
            SP_theta[i + n] = theta[t] - scale * sqrt_P[2][i - 1]

        for i in range(0, 2 * n + 1):
            SP_x[i] = SP_x[i] + delta_t * np.cos(SP_theta[i] + delta_r1)
            SP_y[i] = SP_y[i] + delta_t * np.sin(SP_theta[i] + delta_r1)
            SP_theta[i] = SP_theta[i] + delta_r1 + delta_r2

        Wm = [1 / (2 * n + 2 * lambda_) for _ in range(0, 2 * n + 1)]
        Wm[0] = lambda_ / (n + lambda_)

        mk_x = sum(SP_x[i] * Wm[i] for i in range(0, 2 * n + 1))
        mk_y = sum(SP_y[i] * Wm[i] for i in range(0, 2 * n + 1))
        mk_theta = sum(SP_theta[i] * Wm[i] for i in range(0, 2 * n + 1))
        mk = np.array([mk_x, mk_y, mk_theta]).reshape(3, 1)

        Wc = [1 / (2 * n + 2 * lambda_) for _ in range(0, 2 * n + 1)]
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha ** 2 + beta ** 2)

        P_pred = np.zeros((3, 3))
        for i in range(0, 2 * n + 1):
            Xk = np.array([[SP_x[i]], [SP_y[i]], [SP_theta[i]]])
            P_pred += Wc[i] * (Xk - mk) @ (Xk - mk).T
        P_pred += Q

        sqrt_P = np.linalg.cholesky(P_pred)
        SP_x[0] = mk_x
        SP_y[0] = mk_y
        SP_theta[0] = mk_theta

        for i in range(1, n + 1):
            scale = np.sqrt(n + lambda_)
            SP_x[i] = mk_x + scale * sqrt_P[0][i - 1]
            SP_y[i] = mk_y + scale * sqrt_P[1][i - 1]
            SP_theta[i] = mk_theta + scale * sqrt_P[2][i - 1]

            SP_x[i + n] = mk_x - scale * sqrt_P[0][i - 1]
            SP_y[i + n] = mk_y - scale * sqrt_P[1][i - 1]
            SP_theta[i + n] = mk_theta - scale * sqrt_P[2][i - 1]

        sens = get_sensors(t + 1, sensors)
        m = len(sens)
        if m == 0:
            X.append(mk_x)
            Y.append(mk_y)
            theta.append(mk_theta)
            P = P_pred
            continue

        Yk = [0 for _ in range(2 * n + 1)]
        yk = np.zeros(m)
        for j in range(0, 2 * n + 1):
            pred_r = np.zeros((m, 1))
            for i, s in enumerate(sens):
                r_x, r_y = landmarks[int(s[1] - 1)]
                yk[i] = s[2]
                pred_r[i][0] = np.sqrt((r_x - SP_x[j]) ** 2 + (r_y - SP_y[j]) ** 2)
            Yk[j] = pred_r

        Mu = np.zeros((m, 1))
        for i in range(0, 2 * n + 1):
            Mu += Wm[i] * Yk[i]

        R = np.diag(np.full(m, noise_variance))
        S = np.zeros((m, m))
        for i in range(0, 2 * n + 1):
            S += Wc[i] * (Yk[i] - Mu) @ (Yk[i] - Mu).T
        S += R

        C = np.zeros((3, m))
        for i in range(0, 2 * n + 1):
            X_diff = np.array([[SP_x[i]], [SP_y[i]], [SP_theta[i]]]) - mk
            C += Wc[i] * X_diff @ (Yk[i] - Mu).T

        K = C @ np.linalg.inv(S)
        mk = mk + K @ (yk.reshape(-1, 1) - Mu)
        P = P_pred - K @ S @ K.T

        X.append(mk[0][0])
        Y.append(mk[1][0])
        theta.append(mk[2][0])

    return X, Y


if __name__ == '__main__':
    # Загрузка данных
    landmarks_data = pd.read_csv('data files/landmarks.dat', delimiter=' ', index_col=0).to_numpy()
    sensor_data = pd.read_csv('data files/sensor_data_ekf.dat', delimiter=' ')

    sensor_data['t'] = sensor_data['t'].astype(object)

    t_step = 0
    for index, row in sensor_data.iterrows():
        if row['t'] == 'ODOMETRY':
            t_step += 1
        else:
            sensor_data.loc[index, 't'] = t_step

    # 0 - время, 1 - номер сенсора, 2 - расстояние
    sensors_data = sensor_data[sensor_data['t'] != 'ODOMETRY'].drop(columns=['c']).to_numpy()
    deltas_data = sensor_data[sensor_data['t'] == 'ODOMETRY'].drop(columns=['t'])
    deltas_data.reset_index(drop=True, inplace=True)
    deltas_data = deltas_data.to_numpy()

    # Запуск фильтров
    ekf_x, ekf_y = ekf_localization(deltas_data, sensors_data, landmarks_data)
    ukf_x, ukf_y = ukf_localization(deltas_data, sensors_data, landmarks_data)

    # Отображение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(ekf_x, ekf_y, label="EKF Estimates", marker='+')
    plt.plot(ukf_x, ukf_y, label="UKF Estimates", marker='.')
    plt.legend()
    plt.title('Robot Localization Result: EKF vs UKF')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
