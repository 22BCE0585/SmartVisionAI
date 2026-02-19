import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import pytesseract
import joblib
from PIL import Image
from src.feature_extraction import extract_features
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load saved components
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Define Neural Network Architecture (same as training)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
model.load_state_dict(torch.load("models/nn_model.pth"))
model.eval()

st.title("SmartVision AI - Intelligent Document Analyzer")

uploaded_file = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", width=600)

    # Convert to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # OCR
    text = pytesseract.image_to_string(img)

    st.subheader("Extracted Text (Preview)")
    st.write(text[:500])

    # Feature Extraction
    features = extract_features(text)
    features = features.reshape(1, -1)

    # Scaling
    features_scaled = scaler.transform(features)

    # Convert to tensor
    X_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # -------- Prediction --------
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = label_encoder.inverse_transform([predicted.item()])[0]

    st.subheader("Prediction")
    st.write(f"Document Type: **{predicted_class}**")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")

        # --------- GEN AI PART ---------
    st.subheader("AI Generated Insight")

    with st.spinner("Generating AI summary..."):

        try:
            prompt = f"""
            The following text was extracted from a document classified as {predicted_class}.
            Provide a professional summary and insights about this document.

            Extracted Text:
            {text[:1500]}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an intelligent document analysis assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            summary = response.choices[0].message.content

        except Exception as e:
            # -------- FALLBACK MODE --------
            word_count = len(text.split())
            char_count = len(text)

            summary = f"""
            ⚠ Live LLM service unavailable (Quota/Billing limit).

            Intelligent Local Analysis:

            • Document Type: {predicted_class.upper()}
            • Word Count: {word_count}
            • Character Count: {char_count}
            • Confidence Level: {confidence.item()*100:.2f}%

            This document shows structural characteristics typical of a {predicted_class}.
            The classification was performed using a neural network trained on extracted statistical features.
            """

    st.write(summary)



