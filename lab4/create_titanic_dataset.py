# create_titanic_dataset.py
from catboost.datasets import titanic
import pandas as pd

train, _ = titanic()
train.to_csv("titanic.csv", index=False)
