import pandas as pd

# Загружаем текущую версию датасета
df = pd.read_csv("titanic.csv")

# Преобразуем колонку Sex в one-hot формат
df = pd.get_dummies(df, columns=["Sex"])

# Сохраняем новую версию датасета
df.to_csv("titanic.csv", index=False)
