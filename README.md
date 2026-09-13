# sim-dsp-slam

> Этот репозиторий содержит комплекс лабораторных и практических работ, выполненных в рамках профильных университетских дисциплин.


---

## Структура проекта 
### 1. Оценивание состояния и фильтрация (State Estimation)
* **[Particle Filter (Локализация по датчикам)](./state-estimation/particle-filter-sensor/)** — фильтрация шумов измерений и оценка траектории.
* **[Extended Kalman Filter (EKF)](./state-estimation/extended-kalman-filter/)** — расширенный фильтр Калмана для локализации агента по известным ориентирам.
* **[Unscented Kalman Filter (UKF)](./state-estimation/extended-unscented-kalman-filter/)** — сигма-точечный фильтр для работы с нелинейными моделями движения.
* **[Rao-Blackwellized Particle Filter (RBPF)](./state-estimation/rbpf/)** — фильтр частиц с аналитическим снижением размерности пространства состояний.
* **[Error-State Kalman Filter (ESKF)](./state-estimation/error-state-kalman-filter/)** — применение фильтра ошибок для интеграции данных INS/GPS (навигационных систем).


### 2. Локализация и картирование (SLAM)
* **[Occupancy Grid Mapping](./slam/occupancy-grid-mapping/)** — построение вероятностной сеточной карты проходимости среды по данным дальномеров.
* **[Particle Filter (Локализация по карте)](./state-estimation/particle-filter-map/)** — глобальная локализация мобильного робота в известном пространстве помещения.
* **[EKF-SLAM](./slam/ekf-slam/)** — реализация SLAM на базе расширенного фильтра Калмана.
* **[FastSLAM](./slam/fast-slam/)** — классический алгоритм одновременной локализации и картирования на основе частиц.
* **[GraphSLAM](./slam/graph-slam/)** — графовый SLAM, оптимизирующий траекторию и положение ориентиров через фактор-графы.



---

## Стек технологий и инструменты
* **Языки программирования:** Python 3.12
* **Среда разработки:** PyCharm

---
