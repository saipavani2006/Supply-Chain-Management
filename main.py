from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet import Prophet
from pydantic import BaseModel

# Load cleaned data
df = pd.read_csv('cleaned_supply_chain_data.csv')

app = FastAPI()

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/all_data")
def get_all_data():
    return df.to_dict(orient="records")

@app.get("/forecast/{product_id}")
def forecast_sales(product_id: str):
    product_df = df[df["Product ID"] == product_id]
    if product_df.empty:
        raise HTTPException(status_code=404, detail="Product not found")

    time_series = product_df[['Date', 'Revenue generated']].copy()
    time_series.rename(columns={"Date": "ds", "Revenue generated": "y"}, inplace=True)

    model = Prophet()
    model.fit(time_series)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].tail(30).to_dict(orient="records")

@app.get("/inventory_optimize/{product_id}")
def optimize_inventory(product_id: str):
    product_df = df[df["Product ID"] == product_id]
    if product_df.empty:
        raise HTTPException(status_code=404, detail="Product not found")

    avg_demand = product_df["Order quantities"].mean()
    lead_time = product_df["Lead times"].mean()

    reorder_point = avg_demand * lead_time

    return {
        "product_id": product_id,
        "average_demand": avg_demand,
        "lead_time": lead_time,
        "reorder_point": reorder_point
    }

class MarketText(BaseModel):
    text: str

@app.post("/market_analysis")
def analyze_sentiment(text: str):
    # Mock response for now
    return {
        "sentiment": "positive",
        "confidence": 0.91,
        "suggested_action": "Increase stock for upcoming demand"
    }
