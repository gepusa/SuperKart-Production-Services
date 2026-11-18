
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart")

# Load the trained pipeline
model = joblib.load("superkart_model.joblib")

def preprocess_raw_dataframe(df_raw):
    """Replicates training preprocessing for raw input DataFrames"""
    df = df_raw.copy()

    # 1. Harmonize product sugar categories
    if 'Product_Sugar_Content' in df.columns:
        df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace('reg', 'Regular')

    # 2. Extract Product_Id characters
    if 'Product_Id' in df.columns:
        df['Product_Id_char'] = df['Product_Id'].str[:2]

        # 3. Create Product_Category mapped variable
        category_map = {'FD': 'Food', 'DR': 'Drink', 'NC': 'Non-Consumable'}
        df['Product_Category'] = df['Product_Id_char'].map(category_map)

    # 4. Calculate Store Age (Years)
    if 'Store_Establishment_Year' in df.columns:
        df['Store_Age_Years'] = 2025 - df['Store_Establishment_Year']

    # Drop unnecessary columns
    drop_cols = ['Product_Type', 'Product_Id', 'Store_Establishment_Year', 'Product_Store_Sales_Total']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Explicitly enforce correct training features structure and ordering
    expected_columns = [
        'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_MRP', 'Store_Id', 'Store_Size', 'Store_Location_City_Type',
        'Store_Type', 'Product_Id_char', 'Product_Category', 'Store_Age_Years'
    ]
    return df[expected_columns]

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System - Automated Preprocessing API"

@superkart_api.post('/v1/predict')
def predict_sales():
    # Accept a single raw record JSON request
    data = request.get_json()

    # Transform raw JSON into DataFrame
    raw_df = pd.DataFrame([data])

    try:
        # Clean the input records on-the-fly
        processed_df = preprocess_raw_dataframe(raw_df)
        prediction = model.predict(processed_df).tolist()[0]
        return jsonify({'Sales': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': f"Preprocessing/Prediction failed: {str(e)}"}), 400

@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Accept an uploaded raw CSV file containing arbitrary rows
    file = request.files['file']
    raw_df = pd.read_csv(file)

    try:
        # Automatically clean the raw file batch dynamically
        processed_df = preprocess_raw_dataframe(raw_df)
        predictions = model.predict(processed_df).tolist()

        output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}
        return jsonify(output_dict)
    except Exception as e:
        return jsonify({'error': f"Batch preprocessing/prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    superkart_api.run(debug=True)
