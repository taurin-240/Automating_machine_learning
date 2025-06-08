from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

def run_model(data_path="data/raw/wine.csv", model_out="models/model.joblib"):
    load_dotenv()  # Загружаем переменные из .env

    # Убедимся, что DVC может прочитать ключи из переменных окружения
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy: {acc:.3f}")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)

    return acc

if __name__ == "__main__":
    run_model()
