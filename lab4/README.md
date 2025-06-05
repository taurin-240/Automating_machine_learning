# Обработка данных Titanic с использованием DVC

Этот репозиторий демонстрирует использование DVC (Data Version Control) для отслеживания изменений в датасете Titanic в процессе его пошаговой предварительной обработки и версионирования данных.

---

## ⚙️ Используемые технологии

- Python  
- Git  
- DVC  
- Яндекс Облако (S3-compatible)  
- Pandas  

---

## 📂 Структура проекта

lab4/
├── titanic.csv # Текущая версия датасета
├── titanic.csv.dvc # Файл отслеживания DVC
├── create_titanic_dataset.py # Генерация исходного датасета
├── fill_missing_age.py # Заполнение NaN в колонке Age
├── one_hot_encode_sex.py # One-hot encoding для колонки Sex
README.md

> Примечание: Скрипты и README находятся в корне проекта, а данные и DVC-файлы — в папке `lab4`.

---

## 🗃️ Удалённое хранилище

Удалённое хранилище подключено через Яндекс Облако (S3-compatible) и настроено через DVC:

```bash
dvc remote add -d yandex_s3 s3://dvc-titanic-bucket/dvcstore
dvc remote modify yandex_s3 endpointurl https://storage.yandexcloud.net
dvc remote modify yandex_s3 access_key_id <ACCESS_KEY>
dvc remote modify yandex_s3 secret_access_key <SECRET_KEY>
