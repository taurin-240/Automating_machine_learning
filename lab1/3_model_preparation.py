import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

def load_data(folder):
    """Загружает предобработанные данные из указанной папки."""
    for file in os.listdir(folder):
        if file.startswith("processed_") and file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path, index_col=0)
            
            # Анализ структуры данных: целевая переменная - 'Global_active_power'
            if 'Global_active_power' not in df.columns:
                print("Ошибка: в данных отсутствует целевая переменная 'Global_active_power'")
                return None, None
            
            X = df.drop(columns=['Global_active_power'])  # Все кроме целевой переменной - признаки
            y = df['Global_active_power']  # Целевая переменная
            
            return X, y
    return None, None

def train_and_save_model(train_folder, model_path):
    """Обучает модель и сохраняет её в файл."""
    X_train, y_train = load_data(train_folder)
    if X_train is None or y_train is None:
        print("Ошибка: не удалось загрузить данные для обучения.")
        return
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель сохранена в {model_path}")

if __name__ == "__main__":
    train_folder = "train"
    model_file = "energy_model.pkl"
    
    if os.path.exists(train_folder):
        train_and_save_model(train_folder, model_file)
    else:
        print(f"Папка {train_folder} не найдена.")
