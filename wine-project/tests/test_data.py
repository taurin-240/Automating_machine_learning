import unittest
import pandas as pd
import os


class TestDataQuality(unittest.TestCase):
    """Комплексные тесты качества данных для wine.csv"""

    @classmethod
    def setUpClass(cls):
        """Загрузка данных один раз для всех тестов"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'raw', 'wine.csv')

        # Проверка существования файла
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл данных не найден по пути: {data_path}")

        cls.df = pd.read_csv(data_path)
        cls.class_counts = cls.df['target'].value_counts()

        # Вывод информации о данных
        print("\nЗагружены данные:")
        print(f"Общее количество образцов: {len(cls.df)}")
        print("Распределение классов:")
        print(cls.class_counts.to_string())

    def test_null_values(self):
        """Проверка на отсутствие пропущенных значений"""
        null_columns = self.df.columns[self.df.isnull().any()].tolist()
        self.assertFalse(null_columns, f"Обнаружены пропуски в колонках: {null_columns}")

    def test_dataset_size(self):
        """Проверка минимального размера датасета"""
        self.assertGreaterEqual(len(self.df), 178, "Датасет содержит меньше 178 строк")

    def test_target_presence(self):
        """Проверка наличия целевой переменной"""
        self.assertIn('target', self.df.columns, "Отсутствует колонка 'target'")

    def test_target_classes(self):
        """Проверка количества классов"""
        self.assertEqual(self.df['target'].nunique(), 3, "Должно быть ровно 3 класса")

    def test_target_values(self):
        """Проверка неотрицательных значений target"""
        self.assertTrue((self.df['target'] >= 0).all(), "Обнаружены отрицательные значения")

    def test_class_distribution(self):
        """Проверка распределения классов"""
        min_samples = self.class_counts.min()
        self.assertGreater(min_samples, 5, f"Наименьший класс содержит всего {min_samples} образцов")


if __name__ == "__main__":
    unittest.main(verbosity=2)