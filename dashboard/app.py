import streamlit as st
import pandas as pd
import altair as alt

from scraper.google_shopping import fetch_competitor_prices
from model.predictor import predict_optimal_price
from model.utils import load_inventory_demand

st.set_page_config(page_title="Smart AI Pricing", layout="wide")
st.title("🧠 Smart AI Pricing Dashboard")

# Input
product = st.text_input("🔍 Enter a product name (e.g., OnePlus Nord 5G):")

if product:
    with st.spinner("Fetching competitor prices and computing optimal price..."):
        try:
            # Step 1: Fetch competitor prices
            competitors = fetch_competitor_prices(product)
            if not competitors:
                st.error("No competitor pricing data found.")
            else:
                # Step 2: Load inventory and demand
                inventory, demand = load_inventory_demand(product)

                # Step 3: Predict optimal price
                optimal_price = predict_optimal_price(
                    competitor_data=competitors,
                    inventory=inventory,
                    demand=demand
                )

                # Prepare DataFrame
                df = pd.DataFrame(competitors)
                df["source"] = df["source"].str.title()
                df["price"] = df["price"].astype(float)

                # Add optimal price row
                df_opt = pd.DataFrame([{"source": "Optimal", "price": optimal_price}])
                df_chart = pd.concat([df, df_opt], ignore_index=True)

                # Display data
                st.subheader("📊 Competitor Prices")
                st.dataframe(df)

                st.subheader("📈 Optimal Price Recommendation")
                st.metric(label="💡 Optimal Price", value=f"₹{optimal_price:.2f}")
                st.markdown(f"- **Inventory Level**: `{inventory}`")
                st.markdown(f"- **Demand Score**: `{demand}`")

                # Chart
                chart = alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X('source:N', title='Source'),
                    y=alt.Y('price:Q', title='Price (₹)'),
                    color=alt.Color('source:N', scale=alt.Scale(domain=['Optimal'], range=['#ff7f0e']))
                ).properties(
                    width=600,
                    height=400,
                    title="Competitor vs AI Optimal Price"
                )

                st.altair_chart(chart)

                # CSV download
                st.download_button(
                    "⬇️ Download Optimized Prices CSV",
                    data=df_chart.to_csv(index=False),
                    file_name="optimized_prices.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Sidebar
st.sidebar.markdown("### ⚙️ Controls")
st.sidebar.info("Enter a product name to fetch competitor prices and receive an AI-powered optimal pricing recommendation.")
