import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def generate_data(num_steps, sensor_pos, cycle_radius, num_sensors, mu_0, c_matrix, g_matrix):
    """Генерация истинной траектории диполя и синтетических показаний датчиков."""
    angle = 0.0
    angular_speed = 2 * np.pi / num_steps

    p0 = np.array([cycle_radius * np.cos(angle), cycle_radius * np.sin(angle)])
    q0 = np.array([1.0, 0.5])
    x0 = np.concatenate((p0, q0))

    xk = np.zeros((num_steps + 1, 4))
    xk[0, :] = x0
    e_vector = np.array([0, 0, 1])  # Вектор перпендикулярный плоскости P

    y_measurements = np.zeros((num_steps, num_sensors))

    for k in range(num_steps):
        wk = np.random.multivariate_normal(np.zeros(4), c_matrix)
        angle += angular_speed

        # Движение по окружности для координат (p) и случайное блуждание для момента (q)
        xk[k + 1, :2] = np.array([cycle_radius * np.cos(angle), cycle_radius * np.sin(angle)])
        xk[k + 1, 2:] = xk[k, 2:] + wk[2:]

        b_xk = np.zeros(num_sensors)
        p_curr = xk[k + 1, :2]
        q_curr = xk[k + 1, 2:]

        for j in range(num_sensors):
            r_minus_p = sensor_pos[j, :] - np.array([p_curr[0], p_curr[1], 0.0])
            q_cross_r_minus_p = np.cross(np.array([q_curr[0], q_curr[1], 0.0]), r_minus_p)
            norm_r_minus_p_cubed = np.linalg.norm(r_minus_p) ** 3
            b_xk[j] = (mu_0 / (4 * np.pi)) * (np.dot(e_vector, q_cross_r_minus_p) / norm_r_minus_p_cubed)

        vk = np.random.multivariate_normal(np.zeros(num_sensors), g_matrix)
        y_measurements[k, :] = b_xk + vk

    return xk, y_measurements


def calculate_jacobian_h(sensors_positions, dipole_position, num_sensors, mu_0):
    """Вычисление линейной матрицы наблюдения H для магнитного момента q."""
    h_matrix = np.zeros((num_sensors, 2))
    for j in range(num_sensors):
        p1, p2 = dipole_position
        r1, r2, _ = sensors_positions[j]
        norm_r_minus_p_cubed = np.linalg.norm(sensors_positions[j, :] - np.array([p1, p2, 0.0])) ** 3
        const = (mu_0 / (4 * np.pi)) * (1.0 / norm_r_minus_p_cubed)

        h_matrix[j, 0] = const * (r2 - p2)
        h_matrix[j, 1] = const * (r1 - p1) * (-1.0)

    return h_matrix


def generate_sensor_positions(cycle_radius, num_sensors_axis, height):
    """Создание сетки измерительных сенсоров на высоте height."""
    x = np.linspace(-cycle_radius, cycle_radius, num_sensors_axis)
    y = np.linspace(-cycle_radius, cycle_radius, num_sensors_axis)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    sensors = np.column_stack([
        points,
        np.full((len(points), 1), height)
    ])
    return sensors


def run_rbpf_estimation():
    # --- Инициализация параметров симуляции ---
    height = 1.0
    num_steps = 100
    ax_count_sens = 10
    total_sensors = ax_count_sens * ax_count_sens
    radius = 3.0
    mu_0 = 1.0

    lambda_param, delta_param = 0.1, 0.01

    # Матрицы ковариации шумов
    c_matrix = np.diag([lambda_param ** 2, lambda_param ** 2, delta_param ** 2, delta_param ** 2])
    g_matrix = np.diag([0.0005] * total_sensors)

    # Модель Калмана для скрытого состояния момента (q) внутри частиц
    a_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    q_matrix = np.diag([delta_param ** 2, delta_param ** 2])  # ИСПРАВЛЕНО: Для момента используется delta

    # Генерация сенсорной сетки и синтетических данных
    sens_pos = generate_sensor_positions(cycle_radius=radius + 1, num_sensors_axis=ax_count_sens, height=height)
    xk_true, y_measured = generate_data(num_steps, sens_pos, radius, total_sensors, mu_0, c_matrix, g_matrix)

    # --- Инициализация Фильтра Частиц Рао-Блэквелла (RBPF) ---
    count_particles = 300
    particles = np.random.randn(2, count_particles) * np.random.uniform(-0.25, 0.25, (2, count_particles)) + np.array(
        [[radius], [0.0]])
    particles_weights = np.ones(count_particles) / count_particles

    # Оценки матожидания (m) и ковариации (P) фильтра Калмана для каждой частицы
    m_states = np.zeros((count_particles, 2))
    p_covariances = np.tile(np.eye(2), (count_particles, 1, 1))

    estimated_positions = np.zeros((2, num_steps))

    # --- Основной цикл фильтрации ---
    for k in range(num_steps):
        for i in range(count_particles):
            # 1. Линейный Фильтр Калмана: Этап прогноза для момента (q)
            m_states[i] = a_matrix @ m_states[i]
            p_covariances[i] = a_matrix @ p_covariances[i] @ a_matrix.T + q_matrix

            # 2. Фильтр частиц: Модель движения для координат (p)
            wk1 = np.random.multivariate_normal([0.0, 0.0], np.diag([lambda_param ** 2, lambda_param ** 2]))
            particles[0, i] += wk1[0]
            particles[1, i] += wk1[1]

            # 3. Расчет матрицы связи H и обновление весов частиц через функцию правдоподобия
            h_mat = calculate_jacobian_h(sens_pos, particles[:, i], total_sensors, mu_0)
            s_mat = h_mat @ p_covariances[i] @ h_mat.T + g_matrix

            # Подсчет веса частицы на основе многомерного нормального распределения
            particles_weights[i] = multivariate_normal.pdf(y_measured[k, :], mean=h_mat @ m_states[i], cov=s_mat)

            # 4. Линейный Фильтр Калмана: Этап коррекции для момента (q)
            k_gain = p_covariances[i] @ h_mat.T @ np.linalg.inv(s_mat)
            m_states[i] = m_states[i] + k_gain @ (y_measured[k, :] - h_mat @ m_states[i])
            p_covariances[i] = p_covariances[i] - k_gain @ s_mat @ k_gain.T

        # Нормализация весов
        particles_weights /= np.sum(particles_weights)

        # Оценка текущей позиции диполя (взвешенное среднее)
        estimated_positions[0, k] = np.sum(particles[0] * particles_weights)
        estimated_positions[1, k] = np.sum(particles[1] * particles_weights)

        # Стратегия Ресэмплирования (выбор выживших частиц)
        indices = np.random.choice(np.arange(count_particles), size=count_particles, p=particles_weights)
        particles = particles[:, indices]
        m_states = m_states[indices]
        p_covariances = p_covariances[indices]
        particles_weights = np.ones(count_particles) / count_particles

    # --- Итоговая визуализация результатов ---
    plt.figure(figsize=(8, 8))
    plt.plot(xk_true[:, 0], xk_true[:, 1], c='green', marker='+', label='Истинная траектория (Ground Truth)')
    plt.plot(estimated_positions[0, :], estimated_positions[1, :], c='red', marker='.', label='Оценка RBPF')
    plt.plot(sens_pos[:, 0], sens_pos[:, 1], 'bx', alpha=0.3, label='Сенсоры МЭГ')

    plt.legend()
    plt.xlabel('Координата X')
    plt.ylabel('Координата Y')
    plt.title('Восстановление активности диполя с помощью RBPF')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    run_rbpf_estimation()
