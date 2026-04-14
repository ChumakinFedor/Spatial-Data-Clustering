#include "kmeans.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

static void skip_to_eol(FILE *f) {
    int ch;
    while ((ch = fgetc(f)) != '\n' && ch != EOF) {
    }
}

int main(void) {
    FILE *fin = fopen("data.csv", "r");
    if (fin == NULL) {
        fprintf(stderr, "Error: cannot open input file data.csv\n");
        return 1;
    }

    size_t capacity = 100;
    size_t n_samples = 0;
    double *data = (double *)malloc(capacity * 2U * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Error: memory allocation failed for data\n");
        fclose(fin);
        return 1;
    }

    int first_char = fgetc(fin);
    if (first_char == EOF) {
        fprintf(stderr, "Error: input file data.csv is empty\n");
        free(data);
        fclose(fin);
        return 1;
    }
    if (isalpha((unsigned char)first_char)) {
        skip_to_eol(fin);
    } else {
        if (ungetc(first_char, fin) == EOF) {
            fprintf(stderr, "Error: failed to read input stream\n");
            free(data);
            fclose(fin);
            return 1;
        }
    }

    while (1) {
        double x = 0.0;
        double y = 0.0;
        int label = 0;
        int parsed = fscanf(fin, " %lf , %lf , %d", &x, &y, &label);

        if (parsed == EOF) {
            break;
        }
        if (parsed != 3) {
            fprintf(stderr, "Error: malformed row in data.csv near sample %zu\n", n_samples);
            free(data);
            fclose(fin);
            return 1;
        }

        if (n_samples == capacity) {
            size_t new_capacity = capacity * 2U;
            double *tmp = (double *)realloc(data, new_capacity * 2U * sizeof(double));
            if (tmp == NULL) {
                fprintf(stderr, "Error: memory reallocation failed for data\n");
                free(data);
                fclose(fin);
                return 1;
            }
            data = tmp;
            capacity = new_capacity;
        }

        data[n_samples * 2U + 0U] = x;
        data[n_samples * 2U + 1U] = y;
        n_samples++;

        skip_to_eol(fin);
    }

    fclose(fin);

    if (n_samples == 0) {
        fprintf(stderr, "Error: no valid samples found in data.csv\n");
        free(data);
        return 1;
    }

    printf("Read %zu points\n", n_samples);

    const int k = 3; // Number of clusters
    int *labels = (int *)malloc(n_samples * sizeof(int));
    double *centroids = (double *)malloc((size_t)k * 2U * sizeof(double));
    if (labels == NULL || centroids == NULL) {
        fprintf(stderr, "Error: memory allocation failed for labels/centroids\n");
        free(data);
        free(labels);
        free(centroids);
        return 1;
    }

    double inertia = 0.0;
    KMeansConfig cfg;
    cfg.n_samples = (int)n_samples;
    cfg.n_features = 2;
    cfg.k = k;
    cfg.max_iter = 100;
    cfg.tolerance = 1e-4;
    cfg.data = data;
    cfg.labels = labels;
    cfg.centroids = centroids;
    cfg.inertia = &inertia;

    if (kmeans_run(&cfg) < 0) {
        fprintf(stderr, "Error: kmeans_run failed\n");
        free(data);
        free(labels);
        free(centroids);
        return 1;
    }

    FILE *fout = fopen("results.csv", "w");
    if (fout == NULL) {
        fprintf(stderr, "Error: cannot open output file results.csv\n");
        free(data);
        free(labels);
        free(centroids);
        return 1;
    }

    fprintf(fout, "x,y,pred_label\n");
    for (size_t i = 0; i < n_samples; ++i) {
        fprintf(fout, "%.10f,%.10f,%d\n", data[i * 2U + 0U], data[i * 2U + 1U], labels[i]);
    }

    fclose(fout);

    printf("Inertia (WCSS): %.10f\n", inertia);

    free(data);
    free(labels);
    free(centroids);
    return 0;
}
