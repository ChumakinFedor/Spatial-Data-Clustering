#ifndef KMEANS_H
#define KMEANS_H

typedef struct {
    int n_samples;
    int n_features;
    int k;
    int max_iter;
    double tolerance;
    const double *data;
    int *labels;
    double *centroids;
    double *inertia;
} KMeansConfig;

int kmeans_run(const KMeansConfig *cfg);

#endif
