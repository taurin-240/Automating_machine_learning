import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Предобрабатывает данные."""
    # Преобразуем столбец даты и времени в datetime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Преобразуем числовые столбцы в float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Удаляем строки с пропущенными значениями
    df.dropna(inplace=True)
    
    return df

def load_and_preprocess_data(folder):
    """Загружает, масштабирует и сохраняет данные."""
    scaler = StandardScaler()
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path, index_col=0)  # Используем первый столбец как индекс
            
            df = preprocess_data(df)  # Применяем предобработку
            
            # Пропускаем нечисловые колонки
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            # Сохраняем предобработанный файл
            processed_path = os.path.join(folder, f"processed_{file}")
            df.to_csv(processed_path)
            print(f"Файл обработан и сохранён: {processed_path}")

if __name__ == "__main__":
    for directory in ["train", "test"]:
        if os.path.exists(directory):
            load_and_preprocess_data(directory)
        else:
            print(f"Папка {directory} не найдена.")
