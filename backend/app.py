import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from preprocess_util import preprocess_raw_data_pipeline  # Import helper utility

superkart_api = Flask("SuperKart")

# Load the trained pipeline
model = joblib.load("superkart_model.joblib")

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System - Preprocessed Prediction API"

@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        data = request.get_json()
        raw_df = pd.DataFrame([data])
        processed_df = preprocess_raw_data_pipeline(raw_df)
        prediction = model.predict(processed_df).tolist()[0]
        return jsonify({'Sales': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 400

@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        file = request.files['file']
        raw_df = pd.read_csv(file)
        processed_df = preprocess_raw_data_pipeline(raw_df)
        predictions = model.predict(processed_df).tolist()
        output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}
        return jsonify(output_dict)
    except Exception as e:
        return jsonify({'error': f"Batch prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    superkart_api.run(debug=True)
