from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, Prediction
from fastapi.responses import StreamingResponse
import io
import csv




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model
model = joblib.load("model.pkl")

# Input Schema
class CartData(BaseModel):
    time_spent: int
    cart_value: int
    num_items: int
    logged_in: int
    discount_applied: int
    previous_purchases: int
    device_mobile: int

@app.get("/")
def home():
    return {"message": "CartGuard AI Backend Running"}

@app.post("/predict")
def predict(data: CartData):

    input_data = np.array([[
        data.time_spent,
        data.cart_value,
        data.num_items,
        data.logged_in,
        data.discount_applied,
        data.previous_purchases,
        data.device_mobile
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # ✅ Define risk_level FIRST
    if probability > 0.7:
        risk_level = "High"
    elif probability > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # ✅ Recommended Action
    if probability > 0.7:
        action = "Trigger 15% Discount + Urgency Notification"
    elif probability > 0.4:
        action = "Offer 5% Discount Reminder"
    else:
        action = "No Action Needed"

    # ✅ Explanation Logic
    explanation = []

    if data.time_spent < 5:
        explanation.append("User spent very little time on site.")

    if data.cart_value > 2000:
        explanation.append("High cart value may cause price hesitation.")

    if data.logged_in == 0:
        explanation.append("User is not logged in, lower commitment level.")

    if data.previous_purchases == 0:
        explanation.append("New customer with no purchase history.")

    if not explanation:
        explanation.append("User behavior indicates stable purchase intent.")

    # ✅ Save to database
    db = SessionLocal()
    new_prediction = Prediction(
        time_spent=data.time_spent,
        cart_value=data.cart_value,
        num_items=data.num_items,
        logged_in=data.logged_in,
        discount_applied=data.discount_applied,
        previous_purchases=data.previous_purchases,
        device_mobile=data.device_mobile,
        probability=probability,
        risk_level=risk_level
    )
    db.add(new_prediction)
    db.commit()
    db.close()

    return {
        "abandon_prediction": int(prediction),
        "abandon_probability": float(probability),
        "risk_level": risk_level,
        "recommended_action": action,
        "explanation": explanation
    }


@app.get("/history")
def get_history():
    db = SessionLocal()
    records = db.query(Prediction).all()
    db.close()

    return records

@app.get("/export")
def export_csv():
    db = SessionLocal()
    records = db.query(Prediction).all()
    db.close()

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Cart Value",
        "Time Spent",
        "Probability",
        "Risk Level"
    ])

    # Data
    for record in records:
        writer.writerow([
            record.cart_value,
            record.time_spent,
            round(record.probability * 100, 2),
            record.risk_level
        ])

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cartguard_report.csv"}
    )

