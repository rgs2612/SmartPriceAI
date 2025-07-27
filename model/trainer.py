import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

MODEL_OUTPUT_PATH = os.path.join("model_files", "model.pkl")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate avg/min/max prices and select training features.
    Expects competitor_1_price, competitor_2_price, competitor_3_price, inventory_level, demand_score, base_cost.

    Adds a column 'optimal_price' based on a simple rule (for mock training).
    """
    df["avg_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].mean(axis=1)
    df["min_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].min(axis=1)
    df["max_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].max(axis=1)

    # If 'optimal_price' not in dataset, simulate with a rule
    if "optimal_price" not in df.columns:
        df["optimal_price"] = df["avg_price"] * (1 + 0.05 * df["demand_score"])  # Simple heuristic

    df.rename(columns={
        "inventory_level": "inventory",
        "demand_score": "demand"
    }, inplace=True)

    return df[["avg_price", "min_price", "max_price", "inventory", "demand", "optimal_price"]]

def train_model(merged_data_path: str, save_path: str = MODEL_OUTPUT_PATH):
    """
    Load merged training data, train model, and save it.
    """
    try:
        df_raw = pd.read_csv(merged_data_path)
    except Exception as e:
        print(f"[ERROR] Cannot load training data: {e}")
        return

    df = prepare_features(df_raw)

    X = df[["avg_price", "min_price", "max_price", "inventory", "demand"]]
    y = df["optimal_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ Model trained. Test MAE: {mae:.2f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"💾 Model saved to {save_path}")

if __name__ == "__main__":
    # Example: For local testing
    train_model(merged_data_path="data/merged_data.csv")
