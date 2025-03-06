import os
import pandas as pd
from sklearn.model_selection import train_test_split

def download_dataset():
    """Загружает датасет и возвращает DataFrame."""
    url = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
    zip_file = 'individual+household+electric+power+consumption.zip'
    csv_file = 'household_power_consumption.txt'
    
    # Скачиваем zip-файл
    if not os.path.exists(zip_file):
        os.system(f'wget {url}')
    
    # Извлекаем содержимое zip-файла
    if not os.path.exists(csv_file):
        os.system(f'unzip {zip_file}')
    
    # Загружаем данные в DataFrame
    df = pd.read_csv(csv_file, sep=';', low_memory=False, na_values=['?'])
    print(f"Датасет загружен: {csv_file}")
    return df

def create_directories():
    """Создаёт папки train и test, если они не существуют."""
    for folder in ['train', 'test']:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Папка {folder} создана.")

def split_and_save_data(df, train_folder='train', test_folder='test', test_size=0.2):
    """Разделяет данные на train и test, сохраняет в csv-файлы."""
    create_directories()
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=False)
    
    train_file = os.path.join(train_folder, 'energy_train.csv')
    test_file = os.path.join(test_folder, 'energy_test.csv')
    
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    
    print(f"Train dataset сохранен в {train_file}")
    print(f"Test dataset сохранен в {test_file}")

if __name__ == "__main__":
    df = download_dataset()
    split_and_save_data(df)
