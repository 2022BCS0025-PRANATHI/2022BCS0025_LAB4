import json
import joblib
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "winequality-red.csv"

def main():
    df = pd.read_csv(DATA_PATH, sep=";")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Save trained model
    joblib.dump(model, "model.pkl")

    # REQUIRED by Lab 6
    os.makedirs("app/artifacts", exist_ok=True)

    # Jenkins expects "accuracy"
    metrics = {
        "accuracy": float(r2),
        "mse": float(mse)
    }

    # REQUIRED location
    with open("app/artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete")
    print(metrics)

if __name__ == "__main__":
    main()
