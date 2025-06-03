from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def train_and_save_model():
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Сохраняем модель в текущую директорию
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_and_save_model()
