import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.matplotlib.use('TkAgg')


def get_theta(theta, r, s):
    """
    Функция для генерации тета.
    :param theta: Прошлое значение тета.
    :param r: Радиус.
    :param s: Случайна величина s(k).
    :return: Новое значение тета.
    """
    if theta_bar < theta < 2 * np.pi - theta_bar:
        return (theta + s) % (2 * np.pi)
    if (0 <= theta <= theta_bar) or (2 * np.pi - theta_bar <= theta < 2 * np.pi):
        return (theta + (s / (r + s_bar))) % (2 * np.pi)
    return np.nan


def get_z1(theta, w, r):
    """
    Функция для подсчета z1.
    :param theta: Значение тета.
    :param w: Вес частицы.
    :param r: Радиус.
    :return: Значение z1.
    """
    if theta_bar < theta < 2 * np.pi - theta_bar:
        return L - r * np.cos(theta) + w
    if (0 <= theta <= theta_bar) or (2 * np.pi - theta_bar <= theta < 2 * np.pi):
        return L - r * np.cos(theta_bar) + w
    return np.nan


def get_z2(theta):
    """
    Функция для подсчета z2.
    :param theta: Значение тета.
    :return: Значение z2
    """
    if 0 <= theta < np.pi:
        return 1
    if np.pi <= theta < 2 * np.pi:
        return -1
    return np.nan


def visualization(true_thetas, predict_thetas):
    """
    Функция для построения графика с истинными и предсказанными значениями тета.
    :param true_thetas: Истинные значения тета.
    :param predict_thetas: Предсказанные значения тета.
    :return: None
    """
    indexes_x = [i for i in range(1, 201)]
    plt.plot(indexes_x, predict_thetas, label='Предсказанные значения')
    plt.plot(indexes_x, true_thetas, 'r', label='Истинные значения')
    plt.xlabel('Время')
    plt.ylabel('Позиция')
    plt.title('Реальные и предсказанные значения')
    plt.legend()
    plt.show()


def tri_f(x):
    """
    Функция плотности треугольного распределения.
    :param x: Разница между предсказанным и истинным значениями z1.
    :return: Значение плотности распределения в точке x.
    """
    a, b, c = -w_bar, w_bar, 0
    if x < a:
        return 0.0
    if a <= x < c:
        return (2 * (x - a)) / ((b - a) * (c - a))
    if x == c:
        return 2/(b-a)
    if c < x <= b:
        return (2 * (b - x)) / ((b - a) * (b - c))
    if x > b:
        return 0.0


def new_weight(z_1, z_2, true_z_1, true_z_2):
    """
    Функция для нахождения веса частицы.
    :param z_1: Предсказанное значение z1.
    :param z_2: Предсказанное значение z2.
    :param true_z_1: Истинное значение z1.
    :param true_z_2: Истинное значение z2.
    :return: Вес частицы.
    """
    w1 = tri_f(true_z_1 - z_1)
    if np.isnan(true_z_2):
        return w1*1.0
    if true_z_2 == z_2:
        return w1*1.0
    else:
        return w1*0.0


if __name__ == '__main__':
    # Векторизация функций
    vectorized_get_theta = np.vectorize(get_theta)
    vectorized_get_z1 = np.vectorize(get_z1)
    vectorized_get_z2 = np.vectorize(get_z2)
    vectorized_new_weight = np.vectorize(new_weight)

    # Известные параметры
    N = 5000
    L = 2
    w_bar = 0.1
    theta_bar = np.pi/3
    s_bar = 0.3
    data = pd.read_csv("PF_data.txt", sep=" ")

    # Инициализация
    R = np.random.uniform(0, 2 * L, N)
    particles = np.random.uniform(0, 2 * np.pi, N)
    weights = np.ones(N) / N
    estimated_positions = np.zeros(200)

    for k in range(200):
        true_z1 = data['distSensor'][k]
        true_z2 = data['halfPlaneSensor'][k]
        S = np.random.uniform(-s_bar, s_bar, N)
        # Обновление частиц
        particles = vectorized_get_theta(particles, R, S)
        z1 = vectorized_get_z1(particles, weights, R)
        z2 = vectorized_get_z2(particles)
        # Расчет и нормировка весов
        weights = vectorized_new_weight(z_1=z1, z_2=z2, true_z_1=true_z1, true_z_2=true_z2)
        weights /= np.sum(weights)

        # Прогноз
        estimated_positions[k] = np.sum(particles * weights)
        # Ресэмплинг
        indices = np.random.choice(np.arange(N), size=N, p=weights)
        particles = particles[indices]
        weights = weights[indices]
        R = R[indices]

    visualization(data['GroundTruth'], estimated_positions)
