#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int validate_config(const KMeansConfig *cfg) {
    if (cfg == NULL) {
        return -1;
    }
    if (cfg->k < 1 || cfg->n_samples < 1 || cfg->max_iter < 1) {
        return -2;
    }
    if (cfg->n_features != 2) {
        return -3;
    }
    if (cfg->data == NULL || cfg->labels == NULL || cfg->centroids == NULL) {
        return -4;
    }
    if (cfg->k > cfg->n_samples) {
        return -5;
    }
    return 0;
}

static double squared_distance_2d(const double *a, const double *b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    return dx * dx + dy * dy;
}

static void init_centroids_spread(const KMeansConfig *cfg) {
    const int n = cfg->n_samples;
    const int k = cfg->k;
    const int f = cfg->n_features;

    int first_idx = rand() % n;
    cfg->centroids[0] = cfg->data[first_idx * f + 0];
    cfg->centroids[1] = cfg->data[first_idx * f + 1];

    for (int c = 1; c < k; ++c) {
        int best_idx = 0;
        double best_min_dist = -1.0;

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

        cfg->centroids[c * f + 0] = cfg->data[best_idx * f + 0];
        cfg->centroids[c * f + 1] = cfg->data[best_idx * f + 1];
    }
}

static int farthest_point_from_nearest_centroid(const KMeansConfig *cfg) {
    const int n = cfg->n_samples;
    const int k = cfg->k;
    const int f = cfg->n_features;

    int farthest_idx = 0;
    double max_min_dist = -1.0;

    for (int i = 0; i < n; ++i) {
        const double *p = &cfg->data[i * f];
        double min_dist = squared_distance_2d(p, &cfg->centroids[0]);

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

    return farthest_idx;
}

int kmeans_run(const KMeansConfig *cfg) {
    int ret = validate_config(cfg);
    if (ret < 0) {
        return ret;
    }

    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    const int n = cfg->n_samples;
    const int f = cfg->n_features;
    const int k = cfg->k;

    init_centroids_spread(cfg);

    for (int iter = 0; iter < cfg->max_iter; ++iter) {
        int counts[k];
        double sums[k * f];
        double old_centroids[k * f];
        double shift = 0.0;

        for (int c = 0; c < k; ++c) {
            counts[c] = 0;
            old_centroids[c * f + 0] = cfg->centroids[c * f + 0];
            old_centroids[c * f + 1] = cfg->centroids[c * f + 1];
            sums[c * f + 0] = 0.0;
            sums[c * f + 1] = 0.0;
        }

        for (int i = 0; i < n; ++i) {
            const double *point = &cfg->data[i * f];
            int best_cluster = 0;
            double best_dist = squared_distance_2d(point, &cfg->centroids[0]);

            for (int c = 1; c < k; ++c) {
                double d = squared_distance_2d(point, &cfg->centroids[c * f]);
                if (d < best_dist) {
                    best_dist = d;
                    best_cluster = c;
                }
            }

            cfg->labels[i] = best_cluster;
            counts[best_cluster] += 1;
            sums[best_cluster * f + 0] += point[0];
            sums[best_cluster * f + 1] += point[1];
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                cfg->centroids[c * f + 0] = sums[c * f + 0] / (double)counts[c];
                cfg->centroids[c * f + 1] = sums[c * f + 1] / (double)counts[c];
            }
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                int far_idx = farthest_point_from_nearest_centroid(cfg);
                cfg->centroids[c * f + 0] = cfg->data[far_idx * f + 0];
                cfg->centroids[c * f + 1] = cfg->data[far_idx * f + 1];
                fprintf(stderr, "Warning: empty cluster %d reinit\n", c);
            }
        }

        for (int c = 0; c < k; ++c) {
            shift += squared_distance_2d(&cfg->centroids[c * f], &old_centroids[c * f]);
        }

        if (shift < cfg->tolerance) {
            break;
        }
    }

    if (cfg->inertia != NULL) {
        double wcss = 0.0;

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
