import os

# === Application Mode ===
MOCK_MODE = True  # Only mock data is used. No SerpAPI/API calls.

# === Scraper Behavior ===
DEFAULT_LOCALE = "in"        # Used for mock reference (no real scraping)
DEFAULT_CURRENCY = "INR"
MAX_COMPETITORS = 10         # Max mock competitor prices
SCRAPE_DELAY = 0             # No delay needed in mock mode

# === File Paths ===
DATA_DIR = "data"
MODEL_DIR = "model_files"

INVENTORY_PATH = os.path.join(DATA_DIR, "inventory_demand.csv")
COMPETITOR_PRICE_PATH = os.path.join(DATA_DIR, "competitor_prices.csv")
MERGED_DATA_PATH = os.path.join(DATA_DIR, "merged_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# === API Keys (not used in mock mode) ===
SERPAPI_KEY = "mock_mode_disabled"