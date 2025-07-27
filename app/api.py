from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd

from scraper.google_shopping import fetch_competitor_prices
from model.predictor import predict_optimal_prices
from model.utils import load_inventory_data

app = FastAPI(
    title="Smart Price AI API",
    description="API to fetch competitor prices and calculate optimal price",
    version="1.0"
)

class PriceResponse(BaseModel):
    product_id: int
    product_name: str
    optimal_price: float
    competitor_prices: Dict[str, float]
    inventory: int
    demand_score: float

@app.get("/price", response_model=PriceResponse)
def get_price(product_id: int = Query(..., description="ID of the product")):
    # Load competitor prices from CSV
    df_competitors = fetch_competitor_prices("data/competitor_prices.csv")
    row = df_competitors[df_competitors["product_id"] == product_id]
    
    if row.empty:
        raise HTTPException(status_code=404, detail="Product not found in competitor data.")
    row = row.iloc[0]

    # Extract competitor prices
    competitor_prices = {
        "competitor_1_price": row["competitor_1_price"],
        "competitor_2_price": row["competitor_2_price"],
        "competitor_3_price": row["competitor_3_price"]
    }

    # Load inventory/demand data
    df_inventory = load_inventory_data("data/inventory_demand.csv")
    inv_row = df_inventory[df_inventory["product_id"] == product_id]
    
    if inv_row.empty:
        raise HTTPException(status_code=404, detail="Product not found in inventory data.")
    inv_row = inv_row.iloc[0]

    # Merge into a single row DataFrame
    merged_df = pd.DataFrame([{
        "product_id": product_id,
        "product_name": row["product_name"],
        "competitor_1_price": competitor_prices["competitor_1_price"],
        "competitor_2_price": competitor_prices["competitor_2_price"],
        "competitor_3_price": competitor_prices["competitor_3_price"],
        "inventory_level": inv_row["inventory_level"],
        "demand_score": inv_row["demand_score"],
        "base_cost": inv_row["base_cost"]
    }])

    # Predict
    merged_df["predicted_optimal_price"] = predict_optimal_prices(merged_df)

    return PriceResponse(
        product_id=product_id,
        product_name=row["product_name"],
        optimal_price=merged_df["predicted_optimal_price"].iloc[0],
        competitor_prices=competitor_prices,
        inventory=inv_row["inventory_level"],
        demand_score=inv_row["demand_score"]
    )
