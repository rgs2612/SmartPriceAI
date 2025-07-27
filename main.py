import pandas as pd
from scraper.google_shopping import fetch_competitor_prices
from model.trainer import train_model
from model.predictor import predict_optimal_prices
from model.utils import merge_data

# Paths
SAVE_PATH = "data/competitor_prices.csv"
INVENTORY_PATH = "data/inventory_demand.csv"
MERGED_PATH = "data/merged_data.csv"
MODEL_PATH = "model_files/model.pkl"

def run_pipeline():
    # Step 1: Fetch mock competitor prices (from CSV)
    print("📥 Loading competitor prices...")
    df_prices = fetch_competitor_prices(SAVE_PATH)
    print(f"✅ Loaded {len(df_prices)} rows of competitor pricing data.")

    # Step 2: Load internal inventory + demand data
    print("📦 Loading inventory and demand data...")
    df_inventory = pd.read_csv(INVENTORY_PATH)
    
    # Step 3: Merge datasets
    print("🔗 Merging data...")
    df_merged = merge_data(df_prices, df_inventory)
    df_merged.to_csv(MERGED_PATH, index=False)
    print("✅ Merged data saved to:", MERGED_PATH)

    # Step 4: Train model
    print("🧠 Training pricing model...")
    train_model(merged_data_path=MERGED_PATH, save_path=MODEL_PATH)

    # Step 5: Predict optimal prices
    print("📊 Predicting optimal prices...")
    df_merged["predicted_optimal_price"] = predict_optimal_prices(df_merged)

    # Step 6: Display preview
    print("\n💡 Sample predictions:")
    print(df_merged[["product_id", "product_name", "average_competitor_price", "predicted_optimal_price"]].head())

if __name__ == "__main__":
    run_pipeline()
