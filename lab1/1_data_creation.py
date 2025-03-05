import numpy as np
import pandas as pd
import os

# Создание директорий train и test
def create_directories():
    try:
        os.makedirs("train", exist_ok=True)  # Создаем папку train, если её нет
        os.makedirs("test", exist_ok=True)  # Создаем папку test, если её нет
    except PermissionError:
        print("Ошибка: Нет прав на создание папок")
        return False
    except Exception as e:
        print(f"Ошибка при создании папок: {e}")
        return False
    return True

# Генерация данных о потреблении энергии в умном доме
def generate_energy_data(n_samples=1000, noise_level=5, anomaly_probability=0.02):
    hours = np.arange(n_samples) % 24
    days = np.arange(n_samples) // 24

    # Базовое потребление энергии
    base_consumption = (
            50
            + 30 * np.sin((hours - 8) * (2 * np.pi / 24))
            + 20 * np.sin((hours - 18) * (2 * np.pi / 24))
    )

    # Температура в доме (если температура выше 25°C, потребление увеличивается на 20%)
    temperature = 20 + 10 * np.sin(2 * np.pi * hours / 24)
    temperature_effect = np.where(temperature > 25, 1.2, 1.0)

    # Выходные дни (где 1 — выходной, 0 — рабочий день, в выходной потребление увеличивается на 10%)
    is_weekend = (days % 7 >= 5).astype(int)
    weekend_effect = np.where(is_weekend, 1.1, 1.0)

    # Общее потребление энергии
    energy_consumption = base_consumption * temperature_effect * weekend_effect

    # Шум и аномалии
    noise = np.random.normal(0, noise_level, n_samples)
    energy_consumption += noise

    anomalies = np.random.choice(n_samples, int(anomaly_probability * n_samples), replace=False)
    energy_consumption[anomalies] += np.random.normal(50, 10, len(anomalies))

    # Создание DataFrame
    data = pd.DataFrame({
        'hour': hours,
        'energy_consumption': energy_consumption,
        'temperature': temperature,
        'is_weekend': is_weekend,
        'anomaly': np.isin(np.arange(n_samples), anomalies).astype(int)  # Флаг аномалии
    })

    return data

def save_data_to_csv(data, folder, filename):
    filepath = os.path.join(folder, filename)
    try:
        data.to_csv(filepath, index=False)
        if not os.path.exists(filepath):  # Проверка, что файл создан
            raise FileNotFoundError(f"Файл {filepath} не был создан")
        print(f"Данные успешно сохранены в {filepath}")
        return True
    except Exception as e:
        print(f"Ошибка при сохранении данных в {filepath}: {e}")
        return False


# Основная функция для создания данных
def main():
    if not create_directories():
        return

    try:
        # Генерация данных
        data = generate_energy_data(n_samples=1000)
        if data.empty:
            raise ValueError("Сгенерированные данные пусты")

        # Разделение на обучающие и тестовые данные
        train, test = np.split(data.sample(frac=1), [int(0.8 * len(data))])

        # Сохранение данных с проверкой
        if not save_data_to_csv(train, "train", "energy_data.csv"):
            return  # Прерываем выполнение, если данные не сохранены
        if not save_data_to_csv(test, "test", "energy_data.csv"):
            return  # Прерываем выполнение, если данные не сохранены

        print("Данные успешно сгенерированы и сохранены")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
