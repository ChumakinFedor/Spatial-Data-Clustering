#include "kmeans.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

// Инициализация центроидов для пустых кластеров. 
// Выбираем образец, который находится дальше всего от своего ближайшего центроида, и устанавливаем новый центроид на этот образец.
static void skip_to_eol(FILE *f) {
    int ch;
    while ((ch = fgetc(f)) != '\n' && ch != EOF) {
    }
}

int main(void) {
    FILE *fin = fopen("data.csv", "r");
    if (fin == NULL) {
        fprintf(stderr, "Ошбка: не удалось открыть входной файл data.csv\n");
        return 1;
    }

    // Чтение данных из файла data.csv. Ожидается, что каждая строка содержит x, y и label, разделенные запятыми.
    size_t capacity = 100;
    size_t n_samples = 0;
    double *data = (double *)malloc(capacity * 2U * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Error: memory allocation failed for data\n");
        fclose(fin);
        return 1;
    }
    // Читаем первый символ файла, чтобы определить, есть ли заголовок. Если первый символ - это буква, предполагаем, что это заголовок и пропускаем его.
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
    // Чтение данных из файла, динамически расширяя массив при необходимости. Каждая строка должна содержать x, y и label, разделенные запятыми.
    while (1) {
        double x = 0.0;
        double y = 0.0;
        int label = 0;
        int parsed = fscanf(fin, " %lf , %lf , %d", &x, &y, &label);

        // Если достигнут конец файла, прекращаем чтение. Если строка не соответствует ожидаемому формату, выводим ошибку и завершаем программу.
        if (parsed == EOF) {
            break;
        }
        // Если строка не соответствует ожидаемому формату (не удалось прочитать x, y и label), выводим ошибку и завершаем программу.
        if (parsed != 3) {
            fprintf(stderr, "Ошбка: неправильный формат строки в data.csv около образца %zu\n", n_samples);
            free(data);
            fclose(fin);
            return 1;
        }

        // Если достигнут предел текущей емкости массива, выделяем новый массив с удвоенной емкостью и копируем данные.
        // Если выделение памяти не удалось, выводим ошибку и завершаем программу.
        if (n_samples == capacity) {
            size_t new_capacity = capacity * 2U;
            double *tmp = (double *)realloc(data, new_capacity * 2U * sizeof(double));
            if (tmp == NULL) {
                fprintf(stderr, "Ошбка: не удалось выделить память для данных\n");
                free(data);
                fclose(fin);
                return 1;
            }
            data = tmp;
            capacity = new_capacity;
        }
        // Сохраняем считанные x и y в массив данных. Метка (label) не используется в алгоритме K-Means, но может быть полезна для оценки качества кластеризации.
        data[n_samples * 2U + 0U] = x;
        data[n_samples * 2U + 1U] = y;
        n_samples++;

        skip_to_eol(fin);
    }

    fclose(fin);

    // Если после чтения данных не было найдено ни одного образца, выводим ошибку и завершаем программу.
    if (n_samples == 0) {
        fprintf(stderr, "Ошбка: не найдено ни одного допустимого образца в data.csv\n");
        free(data);
        return 1;
    }

    printf("Read %zu points\n", n_samples); // Выводим количество считанных образцов

    // Настройка конфигурации для алгоритма K-Means. 
    // Устанавливаем количество кластеров (k), максимальное количество итераций, порог для проверки сходимости и указатели на данные, метки и центроиды.
    const int k = 3; // !!!Количество кластеров!!! 

    // Выделение памяти для меток кластеров и центроидов. Метки будут хранить номер кластера для каждого образца, а центроиды будут хранить координаты центров кластеров. 
    // Если выделение памяти не удалось, выводим ошибку и завершаем программу.
    int *labels = (int *)malloc(n_samples * sizeof(int));
    double *centroids = (double *)malloc((size_t)k * 2U * sizeof(double));
    if (labels == NULL || centroids == NULL) {
        fprintf(stderr, "Ошбка: не удалось выделить память для меток/центроидов\n");
        free(data);
        free(labels);
        free(centroids);
        return 1;
    }

    // Инициализация конфигурации для алгоритма K-Means и запуск алгоритма. Если выполнение алгоритма не удалось, выводим ошибку и завершаем программу.
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

    // Запуск алгоритма K-Means с заданной конфигурацией. Если функция возвращает отрицательное значение, это означает, что произошла ошибка, и мы выводим сообщение об ошибке и завершаем программу.
    if (kmeans_run(&cfg) < 0) {
        fprintf(stderr, "Ошбка: не удалось выполнить kmeans_run\n");
        free(data);
        free(labels);
        free(centroids);
        return 1;
    }

    // Сохранение результатов кластеризации в файл results.csv. Каждая строка будет содержать x, y и предсказанную метку кластера для каждого образца. Если не удалось открыть файл для записи, выводим ошибку и завершаем программу.
    FILE *fout = fopen("results.csv", "w");
    if (fout == NULL) {
        fprintf(stderr, "Ошбка: не удалось открыть выходной файл results.csv\n");
        free(data);
        free(labels);
        free(centroids);
        return 1;
    }

    // Записываем заголовок в файл и затем записываем координаты и предсказанные метки для каждого образца. Формат каждой строки: x,y,pred_label. После записи всех данных закрываем файл.
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
