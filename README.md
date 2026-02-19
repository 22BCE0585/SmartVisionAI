# SmartVision AI

SmartVision AI is an end-to-end intelligent document analysis system that combines Computer Vision, Machine Learning, Deep Learning, and Generative AI.

## Features

- OCR-based document text extraction
- Statistical feature engineering
- PCA-based dimensionality analysis
- Logistic Regression & Neural Network classification
- LLM-based intelligent summarization
- Streamlit web deployment

## Tech Stack

- Python
- PyTorch
- OpenCV
- Scikit-learn
- Streamlit
- OpenAI API

## Workflow

Image → OCR → Feature Extraction → Scaling → Neural Network → Prediction → GenAI Summary

## Results

- Logistic Regression Accuracy: ~78%
- Neural Network Accuracy: ~88%

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
streamlit run app.py
