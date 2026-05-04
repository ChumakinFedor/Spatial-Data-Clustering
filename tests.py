#!/usr/bin/env python3
"""
Тестер для проекта K-Means на C
Проверяет: корректность, точность vs sklearn, обработку ошибок, граничные случаи.
Исправлено: учитывает русские сообщения об ошибках из main.c.
"""

import subprocess
import os
import unittest
import numpy as np
import pandas as pd
import shutil
import re
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
C_SOURCE = os.path.join(PROJECT_DIR, "main.c")
BACKUP_SOURCE = os.path.join(PROJECT_DIR, "main.c.bak")
BINARY = os.path.join(PROJECT_DIR, "kmeans_app")
DATA_CSV = os.path.join(PROJECT_DIR, "data.csv")
RESULTS_CSV = os.path.join(PROJECT_DIR, "results.csv")
GEN_SCRIPT = os.path.join(PROJECT_DIR, "generate_data.py")

class TestKMeansProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Генерация данных и подготовка окружения перед всеми тестами"""
        if not os.path.exists(DATA_CSV):
            print("⚙️ Генерация тестовых данных (data.csv)...")
            gen_res = subprocess.run(["python3", GEN_SCRIPT], capture_output=True, text=True)
            if gen_res.returncode != 0:
                raise AssertionError(f"Не удалось запустить generate_data.py:\n{gen_res.stderr}")

        if os.path.exists(C_SOURCE):
            shutil.copy2(C_SOURCE, BACKUP_SOURCE)
        cls._compile(k_val=3)
        print("🔨 Компиляция завершена. Готово к тестированию.\n")

    @classmethod
    def tearDownClass(cls):
        """Восстановление файлов и очистка"""
        if os.path.exists(BACKUP_SOURCE):
            shutil.copy2(BACKUP_SOURCE, C_SOURCE)
            os.remove(BACKUP_SOURCE)
        for f in [BINARY, RESULTS_CSV]:
            if os.path.exists(f): os.remove(f)
        print("\n Очистка завершена.")

    @classmethod
    def _compile(cls, k_val=3):
        """Меняет k в main.c, компилирует, затем возвращает оригинал"""
        if os.path.exists(C_SOURCE):
            with open(C_SOURCE, 'r', encoding='utf-8') as f:
                code = f.read()
            code = re.sub(r'const int k = \d+;', f'const int k = {k_val};', code)
            with open(C_SOURCE, 'w', encoding='utf-8') as f:
                f.write(code)

        res = subprocess.run(
            ["gcc", "-Wall", "-Wextra", "-O2", "-std=c99", "-o", BINARY, "main.c", "kmeans.c", "-lm"],
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        # Восстанавливаем оригинал сразу после компиляции
        if os.path.exists(BACKUP_SOURCE):
            shutil.copy2(BACKUP_SOURCE, C_SOURCE)

        if res.returncode != 0:
            raise AssertionError(f"❌ Компиляция провалилась:\n{res.stderr}")

    def _run_app(self):
        """Запускает бинарник и возвращает результат"""
        return subprocess.run([BINARY], capture_output=True, text=True, cwd=PROJECT_DIR)

    def test_01_basic_run(self):
        """✅ Базовый запуск и проверка артефактов"""
        self.assertTrue(os.path.exists(DATA_CSV), "data.csv отсутствует")
        res = self._run_app()
        self.assertEqual(res.returncode, 0, f"Программа упала:\n{res.stderr}")
        self.assertTrue(os.path.exists(RESULTS_CSV), "results.csv не создан")
        
        df = pd.read_csv(RESULTS_CSV)
        self.assertEqual(len(df), 300, "Ожидается 300 точек")
        self.assertIn("Inertia", res.stdout, "Инерция не выведена")
        
        inertia = float(res.stdout.split(":")[-1].strip())
        self.assertGreater(inertia, 0, "Инерция должна быть > 0")
        print(f"   📊 Inertia: {inertia:.2f} | ✅ Пройден")

    def test_02_accuracy_vs_sklearn(self):
        """✅ Сравнение с эталонной реализацией sklearn"""
        data = pd.read_csv(DATA_CSV)
        res = self._run_app()
        results = pd.read_csv(RESULTS_CSV)
        pred_labels = results["pred_label"].values

        model = KMeans(n_clusters=3, random_state=50, n_init=10)
        model.fit(data[["x", "y"]])
        true_labels = model.labels_

        cm = confusion_matrix(true_labels, pred_labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()

        self.assertGreater(accuracy, 0.90, f"Точность ниже 90%: {accuracy:.2%}")
        print(f"    Точность: {accuracy:.2%} | ✅ Пройден")

    def test_03_k_equals_1(self):
        """✅ Работа при k=1 (все точки в одном кластере)"""
        self._compile(k_val=1)
        res = self._run_app()
        self.assertEqual(res.returncode, 0, f"Программа должна работать при k=1:\n{res.stderr}")
        
        results = pd.read_csv(RESULTS_CSV)
        self.assertTrue((results["pred_label"] == 0).all(), "При k=1 все метки должны быть 0")
        
        inertia = float(res.stdout.split(":")[-1].strip())
        self.assertGreater(inertia, 5000, "Инерция при k=1 должна быть большой")
        print(f"   📊 Inertia (k=1): {inertia:.2f} | ✅ Пройден")

    def test_04_invalid_k_too_large(self):
        """✅ Обработка k > n_samples (должен вернуть ошибку)"""
        self._compile(k_val=500)
        res = self._run_app()
        self.assertNotEqual(res.returncode, 0, "Должен вернуть код ошибки")
        # Проверяем наличие сообщения об ошибке (учитывая опечатку "Ошбка")
        self.assertTrue("Error" in res.stderr or "Ошбка" in res.stderr or "ошбка" in res.stderr, "Нет сообщения об ошибке в stderr")
        print("   ️ Отклонил k > n_samples | ✅ Пройден")

    def test_05_empty_csv(self):
        """✅ Обработка пустого CSV"""
        backup = "data_backup.csv"
        shutil.copy2(DATA_CSV, backup)
        with open(DATA_CSV, "w", encoding="utf-8") as f:
            f.write("x,y,original_label\n")

        res = self._run_app()
        self.assertNotEqual(res.returncode, 0, "Должна быть ошибка при пустом файле")
        # Проверяем русский текст ошибки из main.c
        self.assertTrue(
            "не найдено" in res.stderr.lower() or "empty" in res.stderr.lower(), 
            "Нет упоминания о пустом файле в stderr"
        )
        shutil.copy2(backup, DATA_CSV)
        os.remove(backup)
        print("    Обработал пустой файл | ✅ Пройден")

    def test_06_malformed_csv(self):
        """✅ Обработка битого CSV"""
        backup = "data_backup.csv"
        shutil.copy2(DATA_CSV, backup)
        with open(DATA_CSV, "w", encoding="utf-8") as f:
            f.write("x,y,original_label\n1.0,2.0,0\nbad_data\n")

        res = self._run_app()
        self.assertNotEqual(res.returncode, 0, "Должна быть ошибка при битом CSV")
        # Проверяем русский текст ошибки из main.c
        self.assertTrue(
            "неправильный" in res.stderr.lower() or "malformed" in res.stderr.lower() or "Ошбка" in res.stderr, 
            "Нет ошибки формата в stderr"
        )
        shutil.copy2(backup, DATA_CSV)
        os.remove(backup)
        print("    Обработал битый CSV | ✅ Пройден")

if __name__ == "__main__":
    print("🚀 Запуск тестов K-Means Project...")
    unittest.main(verbosity=2)