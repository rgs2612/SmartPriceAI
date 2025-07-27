import pandas as pd
import numpy as np
import os
from config import INVENTORY_PATH

def clean_price(price_str):
    """
    Clean price string like '₹19,999' or '$199.99' to float.
    """
    if pd.isna(price_str):
        return None
    cleaned = ''.join(ch for ch in str(price_str) if ch.isdigit() or ch == '.')
    try:
        return float(cleaned)
    except ValueError:
        return None

def calculate_demand_score(sales_last_week, avg_sales):
    """
    A mock demand score based on sales performance.
    Scaled 0.0 (low demand) to 1.0 (high demand).
    """
    if avg_sales == 0:
        return 0.0
    score = sales_last_week / avg_sales
    return min(score, 1.0)

def prepare_features(df):
    """
    Add derived features expected by ML model.
    Assumes presence of competitor_1_price, competitor_2_price, competitor_3_price.
    """
    df["avg_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].mean(axis=1)
    df["min_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].min(axis=1)
    df["max_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].max(axis=1)
    df["price_range"] = df["max_price"] - df["min_price"]
    return df

def merge_data(competitor_df, internal_df):
    """
    Merge competitor pricing with internal data (inventory & demand).
    """
    merged = pd.merge(competitor_df, internal_df, on="product_id", how="left")
    
    # Compute features
    merged["avg_price"] = merged[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].mean(axis=1)
    merged["min_price"] = merged[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].min(axis=1)
    merged["max_price"] = merged[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].max(axis=1)
    merged["price_range"] = merged["max_price"] - merged["min_price"]
    
    return merged

def load_inventory_demand(product_name=None):
    """
    Load inventory and demand data from CSV.
    If product_name is provided, return data for that product only.
    """
    try:
        df = pd.read_csv(INVENTORY_PATH)

        if product_name:
            # Get row for specific product
            row = df[df['product_id'] == product_name].iloc[0]
            inventory = int(row["inventory_level"])
            demand = float(row["demand_score"])
            return inventory, demand

        # If no product name, return whole dataframe
        return df
    except Exception as e:
        print(f"Error loading inventory and demand data: {e}")
        return None, None

