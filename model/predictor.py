import joblib
import numpy as np
import os
from typing import List, Optional

from model.optimizer import rule_based_optimizer

MODEL_PATH = os.path.join("model_files", "model.pkl")

def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained ML model from disk.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

def prepare_features(competitor_prices: List[float], inventory: int, demand: float) -> Optional[np.ndarray]:
    """
    Prepare the input features for the ML model.

    Returns:
        np.ndarray: [avg_price, min_price, max_price, inventory, demand]
    """
    if not competitor_prices:
        return None

    avg_price = np.mean(competitor_prices)
    min_price = np.min(competitor_prices)
    max_price = np.max(competitor_prices)

    features = np.array([[avg_price, min_price, max_price, inventory, demand]])
    return features

def predict_optimal_price(
    competitor_data: List[dict],
    inventory: int,
    demand: float,
    base_cost: float = 0
) -> Optional[float]:
    """
    Predict optimal price using ML model, fallback to rule-based logic if model fails.
    
    Parameters:
        competitor_data (list): Each entry has 'source' and 'price'
        inventory (int)
        demand (float)
        base_cost (float): For fallback logic (rule-based)

    Returns:
        float or None
    """
    competitor_prices = [float(row["price"]) for row in competitor_data if "price" in row]

    if not competitor_prices:
        return None

    features = prepare_features(competitor_prices, inventory, demand)
    if features is None:
        return None

    model = load_model()
    if model:
        try:
            predicted_price = model.predict(features)[0]
            return round(predicted_price, 2)
        except Exception as e:
            print(f"[WARNING] Model prediction failed: {e}")

    # Fallback: Rule-based
    print("[INFO] Using rule-based fallback optimizer.")
    return rule_based_optimizer(
        competitor_prices=competitor_prices,
        inventory=inventory,
        demand=demand,
        base_cost=base_cost
    )

def predict_optimal_prices(df):
    """
    Batch prediction for DataFrame with required columns.
    
    Returns:
        DataFrame with added 'predicted_optimal_price' column.
    """
    prices = []
    for _, row in df.iterrows():
        competitor_prices = [
            row["competitor_1_price"],
            row["competitor_2_price"],
            row["competitor_3_price"]
        ]
        predicted = predict_optimal_price(
            competitor_data=[{"price": p} for p in competitor_prices],
            inventory=row["inventory_level"],
            demand=row["demand_score"],
            base_cost=row["base_cost"]
        )
        prices.append(predicted)
    df["predicted_optimal_price"] = prices
    return df
