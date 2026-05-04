#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Проверка корректности конфигурации
static int validate_config(const KMeansConfig *cfg) {
    if (cfg == NULL) {
        return -1; // Указатель на конфигурацию не должен быть NULL
    }
    if (cfg->k < 1 || cfg->n_samples < 1 || cfg->max_iter < 1) {
        return -2; // Количество кластеров, образцов и итераций должно быть положительным
    }
    if (cfg->n_features != 2) {
        return -3; // В данной реализации поддерживается только 2D данные (n_features должно быть равно 2)
    }
    if (cfg->data == NULL || cfg->labels == NULL || cfg->centroids == NULL) {
        return -4; // Указатели на данные, метки и центроиды не должны быть NULL
    }
    if (cfg->k > cfg->n_samples) {
        return -5; // Количество кластеров не может превышать количество образцов
    }
    return 0;
}

// Вычисление квадрата расстояния между двумя точками в 2D пространстве
static double squared_distance_2d(const double *a, const double *b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    return dx * dx + dy * dy;
}

// Инициализация центроидов с помощью метода "spread"
// Выбираем первый центроид случайно, а затем последовательно выбираем следующие центроиды как наиболее удаленные от уже выбранных
static void init_centroids_spread(const KMeansConfig *cfg) {
    const int n = cfg->n_samples;
    const int k = cfg->k;
    const int f = cfg->n_features;

    // Инициализация первого центроида случайным образом
    int first_idx = rand() % n;
    cfg->centroids[0] = cfg->data[first_idx * f + 0];
    cfg->centroids[1] = cfg->data[first_idx * f + 1];

    // Выбор следующих центроидов
    for (int c = 1; c < k; ++c) {
        int best_idx = 0;
        double best_min_dist = -1.0;
        // Для каждого образца вычисляем минимальное расстояние до уже выбранных центроидов и выбираем образец с максимальным минимальным расстоянием
        for (int i = 0; i < n; ++i) {
            const double *p = &cfg->data[i * f];
            double min_dist = squared_distance_2d(p, &cfg->centroids[0]);

            for (int prev = 1; prev < c; ++prev) {
                double d = squared_distance_2d(p, &cfg->centroids[prev * f]);
                if (d < min_dist) {
                    min_dist = d;
                }
            }

            if (min_dist > best_min_dist) {
                best_min_dist = min_dist;
                best_idx = i;
            }
        }
        // Установка нового центроида на выбранный образец
        cfg->centroids[c * f + 0] = cfg->data[best_idx * f + 0];
        cfg->centroids[c * f + 1] = cfg->data[best_idx * f + 1];
    }
}

// Находим индекс образца, который находится дальше всего от своего ближайшего центроида. Это используется для реинициализации пустых кластеров.
static int farthest_point_from_nearest_centroid(const KMeansConfig *cfg) {
    const int n = cfg->n_samples;
    const int k = cfg->k;
    const int f = cfg->n_features;

    int farthest_idx = 0;
    double max_min_dist = -1.0;
    // Для каждого образца вычисляем минимальное расстояние до центроидов и выбираем образец с максимальным минимальным расстоянием
    for (int i = 0; i < n; ++i) {
        const double *p = &cfg->data[i * f];
        double min_dist = squared_distance_2d(p, &cfg->centroids[0]);
        // Вычисляем минимальное расстояние до центроидов
        for (int c = 1; c < k; ++c) {
            double d = squared_distance_2d(p, &cfg->centroids[c * f]);
            if (d < min_dist) {
                min_dist = d;
            }
        }

        if (min_dist > max_min_dist) {
            max_min_dist = min_dist;
            farthest_idx = i;
        }
    }
    // Возвращаем индекс образца, который находится дальше всего от своего ближайшего центроида
    return farthest_idx;
}

// Основная функция для выполнения алгоритма K-Means
int kmeans_run(const KMeansConfig *cfg) {
    // Проверяем корректность конфигурации
    int ret = validate_config(cfg);
    if (ret < 0) {
        return ret;
    }
    // Инициализация генератора случайных чисел (один раз)
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    // Получаем количество образцов, признаков и кластеров из конфигурации
    const int n = cfg->n_samples;
    const int f = cfg->n_features;
    const int k = cfg->k;

    init_centroids_spread(cfg); // Инициализация центроидов с помощью метода "spread"
    
    // Основной цикл алгоритма K-Means
    for (int iter = 0; iter < cfg->max_iter; ++iter) {
        int counts[k]; // Массив для хранения количества образцов в каждом кластере
        double sums[k * f]; // Массив для хранения суммы координат образцов в каждом кластере (для обновления центроидов)
        double old_centroids[k * f]; // Массив для хранения старых координат центроидов (для проверки сходимости)
        double shift = 0.0; // Переменная для хранения общего смещения центроидов в текущей итерации (для проверки сходимости)

        // Инициализация счетчиков, сумм и сохранение старых центроидов
        for (int c = 0; c < k; ++c) {
            counts[c] = 0;
            old_centroids[c * f + 0] = cfg->centroids[c * f + 0];
            old_centroids[c * f + 1] = cfg->centroids[c * f + 1];
            sums[c * f + 0] = 0.0;
            sums[c * f + 1] = 0.0;
        }

        // Назначение образцов к ближайшим центроидам и накопление сумм для обновления центроидов
        for (int i = 0; i < n; ++i) {
            const double *point = &cfg->data[i * f];
            int best_cluster = 0;
            double best_dist = squared_distance_2d(point, &cfg->centroids[0]);
            
            // Находим ближайший центроид для текущего образца
            for (int c = 1; c < k; ++c) {
                double d = squared_distance_2d(point, &cfg->centroids[c * f]);
                if (d < best_dist) {
                    best_dist = d;
                    best_cluster = c;
                }
            }
            // Назначаем образец к ближайшему кластеру и обновляем счетчики и суммы для этого кластера
            cfg->labels[i] = best_cluster;
            counts[best_cluster] += 1;
            sums[best_cluster * f + 0] += point[0];
            sums[best_cluster * f + 1] += point[1];
        }

        // Обновление центроидов на основе накопленных сумм и счетчиков
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                cfg->centroids[c * f + 0] = sums[c * f + 0] / (double)counts[c];
                cfg->centroids[c * f + 1] = sums[c * f + 1] / (double)counts[c];
            }
        }

        // Проверка на пустые кластеры и реинициализация центроидов для пустых кластеров
        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                int far_idx = farthest_point_from_nearest_centroid(cfg);
                cfg->centroids[c * f + 0] = cfg->data[far_idx * f + 0];
                cfg->centroids[c * f + 1] = cfg->data[far_idx * f + 1];
                fprintf(stderr, "Warning: empty cluster %d reinit\n", c);
            }
        }

        // Вычисление общего смещения центроидов для проверки сходимости
        for (int c = 0; c < k; ++c) {
            shift += squared_distance_2d(&cfg->centroids[c * f], &old_centroids[c * f]);
        }
        // Если общее смещение центроидов меньше заданного порога, считаем алгоритм сошедшимся и прекращаем итерации
        if (shift < cfg->tolerance) {
            break;
        }
    }

    // Вычисление инерции (WCSS) при наличии указателя для хранения этого значения
    if (cfg->inertia != NULL) {
        double wcss = 0.0;
        // Для каждого образца вычисляем расстояние до его назначенного центроида и суммируем эти расстояния для получения общего значения инерции (WCSS)
        for (int i = 0; i < n; ++i) {
            const int c = cfg->labels[i];
            if (c < 0 || c >= k) {
                return -6;
            }
            wcss += squared_distance_2d(&cfg->data[i * f], &cfg->centroids[c * f]);
        }

        *cfg->inertia = wcss;
    }

    return 0;
}
