import pandas as pd

# Загружаем текущую версию датасета
df = pd.read_csv("titanic.csv")

# Заполняем пропущенные значения среднего по колонке Age
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Сохраняем модифицированный датасет
df.to_csv("titanic.csv", index=False)

