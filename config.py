import os

# Define base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# File paths
INVENTORY_PATH = os.path.join(DATA_DIR, "inventory_demand.csv")
COMPETITOR_PRICE_PATH = os.path.join(DATA_DIR, "competitor_prices.csv")
MERGED_DATA_PATH = os.path.join(DATA_DIR, "merged_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
