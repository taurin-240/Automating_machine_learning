import os
import pickle
import pandas as pd
from sklearn.metrics import r2_score

def load_data(folder):
    """Загружает предобработанные тестовые данные."""
    for file in os.listdir(folder):
        if file.startswith("processed_") and file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path, index_col=0)
            
            # Проверяем наличие целевой переменной
            if 'Global_active_power' not in df.columns:
                print("Ошибка: в данных отсутствует целевая переменная 'Global_active_power'")
                return None, None
            
            X = df.drop(columns=['Global_active_power'])
            y = df['Global_active_power']
            
            return X, y
    return None, None

def test_model(model_path, test_folder):
    """Загружает модель, тестирует её и выводит метрику R^2."""
    if not os.path.exists(model_path):
        print("Ошибка: файл модели не найден.")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    X_test, y_test = load_data(test_folder)
    if X_test is None or y_test is None:
        print("Ошибка: не удалось загрузить тестовые данные.")
        return
    
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print(f"Model test accuracy is: {accuracy:.4f}")

if __name__ == "__main__":
    model_file = "energy_model.pkl"
    test_folder = "test"
    
    if os.path.exists(test_folder):
        test_model(model_file, test_folder)
    else:
        print(f"Папка {test_folder} не найдена.")
