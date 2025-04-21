import numpy as np
import streamlit as st
import joblib

# Load your trained model and threshold
model = joblib.load("final/logistic_model.pkl")  # Make sure this file is in the same directory
best_threshold = joblib.load("best_threshold.pkl")

# Title
st.title("Product Usefulness Predictor")
st.write("Enter aspect-based sentiment scores for a product to check if it's Useful or Not Useful")

# Input sliders for each aspect
air_flow = st.slider('Air Flow Sentiment', -1.0, 1.0, 0.0, 0.01)
awesome = st.slider('Awesome Sentiment', -1.0, 1.0, 0.0, 0.01)
build_quality = st.slider('Build Quality Sentiment', -1.0, 1.0, 0.0, 0.01)
cooling = st.slider('Cooling Sentiment', -1.0, 1.0, 0.0, 0.01)
design = st.slider('Design Sentiment', -1.0, 1.0, 0.0, 0.01)
noise = st.slider('Noise Sentiment', -1.0, 1.0, 0.0, 0.01)
price = st.slider('Price Sentiment', -1.0, 1.0, 0.0, 0.01)
service = st.slider('Service Sentiment', -1.0, 1.0, 0.0, 0.01)
speed = st.slider('Speed Sentiment', -1.0, 1.0, 0.0, 0.01)

# Compute average sentiment
aspect_scores = [air_flow, awesome, build_quality, cooling, design, noise, price, service, speed]
avg_sentiment = np.mean(aspect_scores)

# Predict using trained model
predicted_prob = model.predict_proba([[avg_sentiment]])[0][1]
label = "Useful" if predicted_prob >= best_threshold else "Not Useful"

# Show prediction
st.subheader("Results")
st.write(f"**Average Sentiment Score:** {avg_sentiment:.4f}")
st.write(f"**Predicted Probability of Usefulness:** {predicted_prob:.4f}")
st.write(f"**Predicted Label:** {label}")
