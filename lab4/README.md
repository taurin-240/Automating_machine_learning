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

- titanic.csv # Текущая версия датасета 
- titanic.csv.dvc # Файл отслеживания DVC
- create_titanic_dataset.py # Генерация исходного датасета
- fill_missing_age.py # Заполнение NaN в колонке Age
- one_hot_encode_sex.py # One-hot encoding для колонки Sex
- README.md


> Примечание: Скрипты и README находятся в корне проекта, а данные и DVC-файлы — в папке `lab4`.

---

## 🗃️ Удалённое хранилище

Удалённое хранилище подключено через Яндекс Облако (S3-compatible) и настроено через DVC:

```bash
dvc remote add -d yandex_s3 s3://dvc-titanic-bucket/dvcstore
dvc remote modify yandex_s3 endpointurl https://storage.yandexcloud.net
dvc remote modify yandex_s3 access_key_id <ACCESS_KEY>
dvc remote modify yandex_s3 secret_access_key <SECRET_KEY>
```

# Titanic Data Processing Pipeline 🚢

Проект демонстрирует полный цикл обработки данных с использованием Git и DVC для контроля версий.

## 🔄 Этапы обработки данных

### 1. 🎯 Выбор признаков
**Действие:** Из исходного датасета оставлены только колонки `Survived`, `Pclass`, `Sex`, `Age`  
**Коммит Git:** `97f3311` (пример)  
**Скрипт:** `select_features.py`

```bash
python scripts/select_features.py
```

### 2. 🧼 Заполнение пропущенных значений
**Действие:** Пропущенные значения (NaN) в колонке Age заполнены средним значением
**Коммит Git:** `ab166cf (пример)
**Скрипт:** fill_missing_age.py

```bash
python scripts/fill_missing_age.py
```

### 3. 🧼 Заполнение пропущенных значений
**Действие:** Категориальный признак Sex преобразован с использованием One-Hot Encoding (созданы признаки Sex_female и Sex_male)
**Коммит Git:** `4382b28 (пример)
**Скрипт:** encode_sex.py

```bash
python scripts/one_hot_encode_sex.py
```

## 🔄 Воспроизведение версий
**Для переключения между версиями датасета:**

```bash
git checkout 97f3311
dvc checkout
```

```bash
git checkout ab166cf
dvc checkout
```

```bash
git checkout 4382b28
dvc checkout
```

**Вернуться на основную ветку:**

```bash
git checkout main
dvc checkout
```
