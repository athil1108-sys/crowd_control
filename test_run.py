
import os
from dotenv import load_dotenv
load_dotenv()

from src.predictor import predict_zone
from src.model import load_model

# Simulated High-Risk Features
high_risk_features = {
    "density": 4.8,
    "velocity": 0.4,
    "rolling_density_mean": 4.2,
    "rolling_velocity_mean": 0.6,
    "density_rate_of_change": 0.5,
    "velocity_rate_of_change": -0.2,
    "density_velocity_ratio": 12.0
}

print("🧠 Running Inference on High-Risk Scenario...")
model, scaler = load_model()
result = predict_zone("Zone_A", high_risk_features, model, scaler)

print("\n--- Prediction Results ---")
print(f"📍 Zone: {result.zone_id}")
print(f"🎯 Probability: {result.risk_probability:.2%}")
print(f"🚦 Risk Level: {result.risk_level.upper()}")
print(f"⏱️  Time to Congestion: {result.time_to_congestion} minutes")
print(f"\n📺 Digital Signage Message (Bedrock):")
print(f"   \"{result.signage_message}\"")
