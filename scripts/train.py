import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "winequality-red.csv"  
# 
def main():
    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, "model.pkl")

    metrics = {"mse": float(mse), "r2": float(r2)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # keep Lab 3 naming too
    with open("results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model.pkl, metrics.json, results.json")
    print(metrics)

if __name__ == "__main__":
    main()
