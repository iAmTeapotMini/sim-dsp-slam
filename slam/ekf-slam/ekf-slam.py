import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Any


def normalize_angle(angle: Any) -> Any:
    """Нормализует угол (или массив углов) к диапазону (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class EKFSLAM:
    """Реализация расширенного фильтра Калмана для задачи SLAM."""

    def __init__(self, landmark_count: int, motion_noise_scale: float = 0.1, measurement_noise_scale: float = 0.01):
        self.m = landmark_count
        self.state_dim = 3 + 2 * self.m

        self.mu = np.zeros((self.state_dim, 1))
        self.sigma = self._init_covariance(value=1e6)

        self.R_t = self._init_covariance(value=motion_noise_scale)
        self.measurement_noise_scale = measurement_noise_scale
        self.F_x = np.hstack([np.eye(3), np.zeros((3, 2 * self.m))])
        self.initialized_landmarks = set()

    def _init_covariance(self, value: float) -> np.ndarray:
        diag = np.zeros(self.state_dim)
        diag[3:] = value
        return np.diag(diag)

    def predict(self, u_t: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Шаг прогноза (Prediction Step)."""
        delta_r1, delta_t, delta_r2 = u_t
        theta = self.mu[2, 0]

        motion_update = np.array([
            [delta_t * np.cos(theta + delta_r1)],
            [delta_t * np.sin(theta + delta_r1)],
            [delta_r1 + delta_r2]
        ])

        mu_pred = self.mu + self.F_x.T @ motion_update
        mu_pred[2, 0] = normalize_angle(mu_pred[2, 0])

        G_x = np.array([
            [1.0, 0.0, -delta_t * np.sin(theta + delta_r1)],
            [0.0, 1.0, delta_t * np.cos(theta + delta_r1)],
            [0.0, 0.0, 1.0]
        ])

        G_t = np.eye(self.state_dim)
        G_t[0:3, 0:3] = G_x

        sigma_pred = G_t @ self.sigma @ G_t.T + self.R_t
        return mu_pred, sigma_pred

    def update(self, mu_pred: np.ndarray, sigma_pred: np.ndarray, measurements: List[List[float]]) -> None:
        """Шаг коррекции (Correction Step)."""
        num_measurements = len(measurements)
        if num_measurements == 0:
            self.mu, self.sigma = mu_pred, sigma_pred
            return

        Q_t = np.eye(2 * num_measurements) * self.measurement_noise_scale
        H_list = []
        z = np.zeros((2 * num_measurements, 1))
        z_hat = np.zeros((2 * num_measurements, 1))

        for idx, (landmark_id, r, phi) in enumerate(measurements):
            landmark_id = int(landmark_id)

            if landmark_id not in self.initialized_landmarks:
                self.initialized_landmarks.add(landmark_id)
                mu_pred[2 * landmark_id + 1, 0] = mu_pred[0, 0] + r * np.cos(phi + mu_pred[2, 0])
                mu_pred[2 * landmark_id + 2, 0] = mu_pred[1, 0] + r * np.sin(phi + mu_pred[2, 0])

            delta_x = mu_pred[2 * landmark_id + 1, 0] - mu_pred[0, 0]
            delta_y = mu_pred[2 * landmark_id + 2, 0] - mu_pred[1, 0]
            q = delta_x ** 2 + delta_y ** 2
            sqrt_q = np.sqrt(q)

            z[idx * 2: idx * 2 + 2] = np.array([[r], [phi]])
            z_hat[idx * 2: idx * 2 + 2] = np.array(
                [[sqrt_q], [normalize_angle(np.arctan2(delta_y, delta_x) - mu_pred[2, 0])]])

            H_j = (1.0 / q) * np.array([
                [-sqrt_q * delta_x, -sqrt_q * delta_y, 0.0, sqrt_q * delta_x, sqrt_q * delta_y],
                [delta_y, -delta_x, -q, -delta_y, delta_x]
            ])

            H_ij = np.zeros((2, self.state_dim))
            H_ij[:, 0:3] = H_j[:, 0:3]
            H_ij[:, 2 * landmark_id + 1: 2 * landmark_id + 3] = H_j[:, 3:5]
            H_list.append(H_ij)

        H = np.vstack(H_list)
        K = sigma_pred @ H.T @ np.linalg.inv(H @ sigma_pred @ H.T + Q_t)

        innovation = z - z_hat
        innovation[1::2] = normalize_angle(innovation[1::2])

        self.mu = mu_pred + K @ innovation
        self.sigma = (np.eye(self.state_dim) - K @ H) @ sigma_pred


def load_dataset(sensor_path: str, landmarks_path: str) -> Tuple[
    List[List[float]], Dict[int, List[List[float]]], np.ndarray]:
    """Считывает и парсит файлы данных."""
    with open(sensor_path, 'r') as f:
        raw_sensor_data = [line.strip().split() for line in f if line.strip()]

    with open(landmarks_path, 'r') as f:
        landmarks = np.array([line.strip().split() for line in f if line.strip()], dtype=float)

    odometry = []
    sensor_measurements = defaultdict(list)
    time_step = -1

    for row in raw_sensor_data:
        if row[0] == 'ODOMETRY':
            odometry.append([float(x) for x in row[1:4]])
            time_step += 1
        elif row[0] == 'SENSOR':
            sensor_measurements[time_step].append([int(row[1]), float(row[2]), float(row[3])])

    return odometry, sensor_measurements, landmarks


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

    plt.title('Результат работы EKF SLAM', fontsize=14)
    plt.xlabel('Координата X (м)', fontsize=12)
    plt.ylabel('Координата Y (м)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    LANDMARK_COUNT = 9

    odometry_data, sensor_data, true_landmarks = load_dataset('sensor_data.dat', 'landmarks.dat')
    slam = EKFSLAM(landmark_count=LANDMARK_COUNT)

    # Массив для хранения истории позиций робота (X, Y)
    robot_trajectory = []

    for t in range(len(odometry_data)):
        mu_p, sigma_p = slam.predict(odometry_data[t])
        slam.update(mu_p, sigma_p, sensor_data[t])

        # Сохраняем текущую предсказанную позицию робота (первые две координаты mu)
        robot_trajectory.append([slam.mu[0, 0], slam.mu[1, 0]])

    robot_trajectory = np.array(robot_trajectory)

    # Вывод результатов в консоль
    print(f"{'ID':<5} | {'True Location':<15} | {'Estimated Location':<20} | {'Error Vector':<15}")
    print("-" * 65)
    for i, landmark in enumerate(true_landmarks):
        l_id = int(landmark[0])
        true_coords = landmark[1:3]
        estimated_coords = slam.mu[3 + i * 2: 3 + i * 2 + 2, 0]
        error = np.abs(true_coords - estimated_coords)
        print(f"{l_id:<5} | {str(true_coords):<15} | {np.round(estimated_coords, 3)} | {np.round(error, 3)}")

    # Отрисовка графиков
    plot_results(robot_trajectory, true_landmarks, slam.mu)
