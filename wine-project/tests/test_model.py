import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from joblib import load
import pandas as pd
import os

class TestModelPerformance(unittest.TestCase):
    """Тесты производительности модели"""

    @classmethod
    def setUpClass(cls):
        """Подгрузка и разбиение данных"""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'wine.csv')
        df = pd.read_csv(data_path)

        X = df.drop(columns=['target'])
        y = df['target']

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        cls.model = RandomForestClassifier(random_state=42)
        cls.model.fit(cls.X_train, cls.y_train)

    def test_accuracy_above_threshold(self):
        """Проверка, что точность выше 85%"""
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(acc, 0.85, f"Точность модели слишком низкая: {acc:.2f}")

    def test_f1_macro_score(self):
        """Проверка F1-метрики (macro)"""
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        self.assertGreaterEqual(f1, 0.85, f"F1 (macro) ниже 0.85: {f1:.2f}")

    def test_prediction_shape(self):
        """Размерность предсказания должна совпадать с тестом"""
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test), "Размерности предсказаний и y_test не совпадают")

    def test_model_predict_range(self):
        """Предсказания модели должны быть только в допустимом диапазоне классов"""
        y_pred = self.model.predict(self.X_test)
        unique_preds = set(y_pred)
        allowed_classes = set(self.y_test.unique())
        self.assertTrue(unique_preds.issubset(allowed_classes),
                        f"Предсказаны недопустимые классы: {unique_preds - allowed_classes}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
