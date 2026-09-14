import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Any


def normalize_angle(angle: Any) -> Any:
    """Нормализует угол (или массив углов) к диапазону (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def init_covariance(length: int, value: float = 1e6) -> np.ndarray:
    """Инициализирует диагональную матрицу ковариации для вектора состояний."""
    diagonal = np.zeros(length)
    diagonal[3:] = value
    return np.diag(diagonal)


def extract_data(raw_data: List[List[str]]) -> Tuple[List[List[float]], Dict[int, List[List[float]]]]:
    """Парсит сырые данные датчиков, разделяя одометрию и измерения."""
    odometry = []
    sensor_measurements = defaultdict(list)
    time_step = -1

    for row in raw_data:
        if row[0] == 'ODOMETRY':
            odometry.append([float(row[1]), float(row[2]), float(row[3])])
            time_step += 1
        elif row[0] == 'SENSOR':
            sensor_measurements[time_step].append([int(row[1]), float(row[2]), float(row[3])])

    return odometry, sensor_measurements


def run_ekf_slam(odometry: List[List[float]], sensors: Dict[int, List[List[float]]],
                 sens_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Выполняет алгоритм EKF-SLAM по всему логу данных.

    Returns:
        robot_trajectory: Траектория робота Nx2 (координаты X, Y)
        mu: Финальный вектор состояний (робот + карта ориентиров)
        Sigma: Финальная матрица ковариации
    """
    state_dim = 3 + 2 * sens_count

    # Матрицы шума процесса и измерений
    R_t = init_covariance(state_dim, value=0.1)
    measurement_noise_scale = 0.01

    # Матрица проекции F_x
    F_x = np.hstack([np.eye(3), np.zeros((3, 2 * sens_count))])

    # Инициализация фильтра Калмана
    init_sensors = []
    mu = np.zeros((state_dim, 1))
    Sigma = init_covariance(state_dim)

    # История перемещений для отрисовки графика
    robot_trajectory = []

    # Основной цикл EKF-SLAM
    for t in range(len(odometry)):
        delta_r1, delta_t, delta_r2 = odometry[t]
        theta = mu[2, 0]

        # --- Шаг прогноза (Prediction) ---
        motion_update = np.array([
            [delta_t * np.cos(theta + delta_r1)],
            [delta_t * np.sin(theta + delta_r1)],
            [delta_r1 + delta_r2]
        ])

        mu_pred = mu + F_x.T @ motion_update
        mu_pred[2, 0] = normalize_angle(mu_pred[2, 0])

        G_xt = np.array([
            [1.0, 0.0, -delta_t * np.sin(theta + delta_r1)],
            [0.0, 1.0, delta_t * np.cos(theta + delta_r1)],
            [0.0, 0.0, 1.0]
        ])

        G_t = np.eye(state_dim)
        G_t[0:3, 0:3] = G_xt

        Sigma_pred = G_t @ Sigma @ G_t.T + R_t

        # --- Шаг коррекции (Update) ---
        m = len(sensors[t])
        if m == 0:
            mu, Sigma = mu_pred, Sigma_pred
            robot_trajectory.append([mu[0, 0], mu[1, 0]])
            continue

        Q_t = np.eye(2 * m) * measurement_noise_scale
        H_list = []
        z = np.zeros((2 * m, 1))
        z_cap = np.zeros((2 * m, 1))

        for index, sensor in enumerate(sensors[t]):
            num_sensor, r, fi = sensor
            num_sensor = int(num_sensor)

            # Инициализация нового ориентира на карте, если встречен впервые
            if num_sensor not in init_sensors:
                init_sensors.append(num_sensor)
                mu_pred[2 * num_sensor + 1, 0] = mu_pred[0, 0] + r * np.cos(fi + mu_pred[2, 0])
                mu_pred[2 * num_sensor + 2, 0] = mu_pred[1, 0] + r * np.sin(fi + mu_pred[2, 0])

            delta_x = mu_pred[2 * num_sensor + 1, 0] - mu_pred[0, 0]
            delta_y = mu_pred[2 * num_sensor + 2, 0] - mu_pred[1, 0]
            q = delta_x ** 2 + delta_y ** 2
            sqrt_q = np.sqrt(q)

            z[index * 2: index * 2 + 2] = np.array([[r], [fi]])
            z_cap[index * 2: index * 2 + 2] = np.array(
                [[sqrt_q], [normalize_angle(np.arctan2(delta_y, delta_x) - mu_pred[2, 0])]])

            # Вычисление низкоразмерного Якобиана H_j
            H_j = (1.0 / q) * np.array([
                [-sqrt_q * delta_x, -sqrt_q * delta_y, 0.0, sqrt_q * delta_x, sqrt_q * delta_y],
                [delta_y, -delta_x, -q, -delta_y, delta_x]
            ])

            # Масштабирование до полной размерности вектора состояний H_ij
            H_ij = np.zeros((2, state_dim))
            H_ij[:, 0:3] = H_j[:, 0:3]
            H_ij[:, 2 * num_sensor + 1: 2 * num_sensor + 3] = H_j[:, 3:5]
            H_list.append(H_ij)

        H = np.vstack(H_list)

        # Коэффициент Калмана (Kalman Gain)
        K_i = Sigma_pred @ H.T @ np.linalg.inv(H @ Sigma_pred @ H.T + Q_t)

        # Инновация (невязка) и нормализация её угловых компонент
        innovation = z - z_cap
        innovation[1::2] = normalize_angle(innovation[1::2])

        # Обновление состояния и ковариации
        mu = mu_pred + K_i @ innovation
        Sigma = (np.eye(state_dim) - K_i @ H) @ Sigma_pred

        # Сохраняем текущую позицию робота
        robot_trajectory.append([mu[0, 0], mu[1, 0]])

    return np.array(robot_trajectory), mu, Sigma


def plot_results(trajectory: np.ndarray, true_landmarks: np.ndarray, estimated_mu: np.ndarray):
    """Строит 2D график траектории робота и положения ориентиров."""
    plt.figure(figsize=(10, 8))

    # 1. Траектория робота
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Предсказанная траектория робота', color='blue', linewidth=2)
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', marker='o', s=100, label='Старт')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', marker='X', s=100, label='Финиш')

    # 2. Истинные ориентиры (Landmarks)
    plt.scatter(true_landmarks[:, 1], true_landmarks[:, 2], color='black', marker='*', s=150,
                label='Истинные ориентиры')
    for landmark in true_landmarks:
        plt.text(landmark[1] + 0.1, landmark[2] + 0.1, f"ID {int(landmark[0])}", fontsize=10, weight='bold')

    # 3. Оцененные ориентиры из вектора состояний (mu)
    est_landmarks_x = estimated_mu[3::2, 0]
    est_landmarks_y = estimated_mu[4::2, 0]
    plt.scatter(est_landmarks_x, est_landmarks_y, color='orange', marker='x', s=100, label='Оцененные ориентиры (SLAM)')

    plt.title('Результат работы EKF SLAM', fontsize=14, weight='bold')
    plt.xlabel('Координата X (м)', fontsize=12)
    plt.ylabel('Координата Y (м)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    LANDMARK_COUNT = 9

    # Чтение файлов данных
    with open('sensor_data.dat', 'r') as f:
        sensor_data_lines = [line.strip().split() for line in f if line.strip()]

    with open('landmarks.dat', 'r') as f:
        landmarks_data = np.array([line.strip().split() for line in f if line.strip()], dtype=float)

    # Предварительная обработка логов
    odometry_log, sensor_log = extract_data(sensor_data_lines)

    # --- Запуск функции EKF-SLAM ---
    robot_path, final_mu, final_Sigma = run_ekf_slam(odometry_log, sensor_log, sens_count=LANDMARK_COUNT)

    # Вывод результатов оценки в консоль
    print(f"\n{'ID':<5} | {'True Location':<15} | {'Estimated Location':<20} | {'Error Vector':<15}")
    print("-" * 65)
    for i, l in enumerate(landmarks_data):
        l_id = int(l[0])
        true_coords = l[1:3]
        estimated_coords = final_mu[3 + i * 2: 3 + i * 2 + 2, 0]
        error = np.abs(true_coords - estimated_coords)
        print(f"{l_id:<5} | {str(true_coords):<15} | {np.round(estimated_coords, 3)} | {np.round(error, 3)}")

    # Отрисовка 2D графика
    plot_results(robot_path, landmarks_data, final_mu)
