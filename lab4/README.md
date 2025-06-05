Обработка данных Titanic с использованием DVC
Этот репозиторий демонстрирует использование DVC (Data Version Control) для отслеживания изменений в датасете Titanic в процессе его пошаговой предварительной обработки и версионирования данных.
Структура проекта
lab4/
├── titanic.csv                # Текущая версия датасета
├── titanic.csv.dvc            # Файл отслеживания DVC
scripts/
├── create_titanic_dataset.py  # Генерация исходного датасета
├── fill_missing_age.py        # Заполнение NaN в колонке Age
├── one_hot_encode_sex.py      # One-hot encoding для колонки Sex
README.md
