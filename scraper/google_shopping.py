import pandas as pd
import os

def fetch_competitor_prices(csv_path="data/competitor_prices.csv") -> pd.DataFrame:
    """
    Simulates scraping competitor prices by reading from a mock CSV.

    Args:
        csv_path (str): Path to the mock CSV file.

    Returns:
        pd.DataFrame: DataFrame with product_id, product_name, and competitor prices.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_cols = {"product_id", "product_name", "competitor_1_price", "competitor_2_price", "competitor_3_price"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing expected columns in CSV: {expected_cols - set(df.columns)}")

    return df

# Example usage for testing
if __name__ == "__main__":
    df = fetch_competitor_prices()
    print(df.head())
