import pickle
import numpy as np
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def get_lidar_data(current_time, lt, threshold=0.1):
    """Поиск индекса измерения Лидара, попавшего в окно синхронизации."""
    for i, time in enumerate(lt):
        if abs(current_time - time) <= threshold:
            return i
    return None


def get_gnss_data(current_time, gnt, threshold=0.1):
    """Поиск индекса измерения GNSS, попавшего в окно синхронизации."""
    # ИСПРАВЛЕНО: Теперь цикл идет строго по массиву gnt (GNSS), а не по лидару
    for i, time in enumerate(gnt):
        if abs(current_time - time) <= threshold:
            return i
    return None


def quaternion_multiply(q1, q2):
    """Умножение двух кватернионов в формате [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([x, y, z, w])


def Rot(q):
    """Преобразование кватерниона [x, y, z, w] в матрицу поворота DCM."""
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q[3], q[0], q[1], q[2]
    R = np.array([
        [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]
    ])
    return R


# Оригинальная динамическая загрузка модулей, требуемая файлом data.py
module_utils = SourceFileLoader("utils", "data/utils.py").load_module()
import data.utils

module_data = SourceFileLoader("data", "data/data.py").load_module()
import data


def run_eskf_simulation():
    """Основная функция для запуска интеграции IMU и фильтрации ESKF."""

    # --- Загрузка бинарных данных симуляции ---
    with open('data/data.pkl', 'rb') as f:
        data_loaded = pickle.load(f)

    gt = data_loaded['gt']
    imu_f = data_loaded['imu_f']
    imu_w = data_loaded['imu_w']
    gnss = data_loaded['gnss']
    lidar = data_loaded['lidar']

    # Параметры калибровки Лидара
    C = np.array([[0.99376, -0.09722, 0.05466],
                  [0.09971, 0.99401, -0.04475],
                  [-0.04998, 0.04992, 0.9975]])
    t = np.array([[0.5], [0.1], [0.5]])

    corrected_lidar = np.array([(np.dot(C, row.reshape(3, 1)) + t).reshape(1, 3)[0] for row in lidar.data])

    # Дисперсии шумов датчиков
    sgm2_acc = 0.01
    sgm2_gyro = 0.01
    sgm2_gnss = 1.0
    sgm2_lidar = 1.0

    g = np.array([[0.0], [0.0], [-9.81]])
    I = np.eye(3)
    L = np.zeros((9, 6))
    L[3:6, 0:3] = I
    L[6:9, 3:6] = I
    P = np.eye(9)

    gnss_timestamps = gnss.t
    lidar_timestamps = lidar.t

    # Инициализация начального состояния по Ground Truth
    r0 = Rotation.from_euler('zyx', np.array([gt.r[0, 2], gt.r[0, 1], gt.r[0, 0]]), degrees=True).as_quat()
    xk = {
        'p': np.array([gt.p[0, 0], gt.p[0, 1], gt.p[0, 2]]).reshape(3, 1),
        'v': np.array([gt.v[0, 0], gt.v[0, 1], gt.v[0, 2]]).reshape(3, 1),
        'q': r0
    }
    xk['q'] = xk['q'] / np.linalg.norm(xk['q'])

    prev_timestamp = imu_f.t[0]
    Pos = [xk['p'].reshape(1, 3)[0]]

    # --- Основной цикл фильтрации ---
    for k, _ in enumerate(imu_f.data):
        timestamp = imu_f.t[k]
        delta_t = timestamp - prev_timestamp
        prev_timestamp = timestamp

        wk_1 = imu_w.data[k].reshape(1, 3)
        fk_1 = imu_f.data[k].reshape(3, 1)

        # 1. Прогноз номинального состояния (Интеграция IMU)
        R_q = Rot(xk['q'])
        xk['p'] = xk['p'] + delta_t * xk['v'] + 0.5 * delta_t ** 2 * (np.dot(R_q, fk_1) + g)
        xk['v'] = xk['v'] + delta_t * (np.dot(R_q, fk_1) + g)
        xk['q'] = quaternion_multiply(xk['q'], Rotation.from_rotvec(wk_1[0] * delta_t).as_quat())
        xk['q'] = xk['q'] / np.linalg.norm(xk['q'])

        # 2. Прогноз матрицы ковариации ошибки P
        Rf = np.dot(R_q, fk_1).T[0]
        a_x = np.array([[0.0, -Rf[2], Rf[1]],
                        [Rf[2], 0.0, -Rf[0]],
                        [-Rf[1], Rf[0], 0.0]])

        F = np.zeros((9, 9))
        F[0:3, 0:3] = I
        F[0:3, 3:6] = I * delta_t
        F[3:6, 3:6] = I
        F[6:9, 6:9] = I
        F[3:6, 6:9] = -1.0 * a_x * delta_t

        G = np.diag([sgm2_acc, sgm2_acc, sgm2_acc,
                     sgm2_gyro, sgm2_gyro, sgm2_gyro]) * delta_t ** 2
        P_pred = np.dot(np.dot(F, P), F.T) + np.dot(np.dot(L, G), L.T)

        # 3. Синхронизация асинхронных датчиков
        lidar_exists = get_lidar_data(timestamp, lidar_timestamps)
        gnss_exists = get_gnss_data(timestamp, gnss_timestamps)

        R_meas, yk, H = None, None, None
        if lidar_exists is not None and gnss_exists is not None:
            H = np.zeros((6, 9))
            H[0:3, 0:3] = I
            H[3:6, 0:3] = I
            R_meas = np.zeros((6, 6))
            R_meas[0:3, 0:3] = I * sgm2_gnss
            R_meas[3:6, 3:6] = I * sgm2_lidar
            yk = np.hstack([gnss.data[gnss_exists], corrected_lidar[lidar_exists]])
        elif lidar_exists is not None:
            H = np.zeros((3, 9))
            H[0:3, 0:3] = I
            R_meas = I * sgm2_lidar
            yk = corrected_lidar[lidar_exists]
        elif gnss_exists is not None:
            H = np.zeros((3, 9))
            H[0:3, 0:3] = I
            R_meas = I * sgm2_gnss
            yk = gnss.data[gnss_exists]
        else:
            print(f"index: {k}, timestamp: {timestamp}")
            print(f"lidar_data: {lidar_timestamps}")
            print(f"gnns_data: {gnss_timestamps}")
            break

        # Вычисление коэффициента усиления Калмана и коррекция
        S = np.dot(np.dot(H, P_pred), H.T) + R_meas
        K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))

        state_extracted = np.hstack([xk['p'].reshape(1, 3)[0], xk['v'].reshape(1, 3)[0], xk['q'][0:3]])
        delta_xk = np.dot(K, (yk - np.dot(H, state_extracted)))
        P = P_pred - np.dot(np.dot(K, S), K.T)

        # 4. Вливание ошибки в номинальное состояние (Инжекция)
        xk['p'] = xk['p'] + delta_xk[0:3].reshape(3, 1)
        xk['v'] = xk['v'] + delta_xk[3:6].reshape(3, 1)
        xk['q'] = quaternion_multiply(Rotation.from_rotvec(delta_xk[6:9]).as_quat(), xk['q'])
        xk['q'] = xk['q'] / np.linalg.norm(xk['q'])

        Pos.append(xk['p'].reshape(1, 3)[0])

    # --- 3D Визуализация результатов ---
    Pos = np.array(Pos)
    gt_fig = plt.figure(figsize=(10, 7))
    ax = gt_fig.add_subplot(111, projection='3d')
    ax.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], c='green', label="Истинная траектория (Ground Truth)")
    ax.plot(Pos[:, 0], Pos[:, 1], Pos[:, 2], c='orange', label="Оценка ESKF")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Результат работы ESKF: Траектория в 3D')
    ax.legend()
    ax.set_zlim(-1, 5)
    plt.show()


if __name__ == '__main__':
    run_eskf_simulation()
