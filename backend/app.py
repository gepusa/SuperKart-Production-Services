# Import necessary libraries
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
superkart_api = Flask("SuperKart")

# Load the trained model pipeline
model = joblib.load("superkart_model.joblib")

def preprocess_raw_dataframe(df_raw):
    df = df_raw.copy()
    
    # 1. Extract Product_Id_char if Product_Id exists
    if 'Product_Id' in df.columns:
        df['Product_Id_char'] = df['Product_Id'].astype(str).str[:2]
    elif 'Product_Id_char' not in df.columns:
        df['Product_Id_char'] = 'FD'
        
    # 2. Map Product_Id_char to Product_Category
    category_map = {'FD': 'Food', 'DR': 'Drink', 'NC': 'Non-Consumable'}
    df['Product_Category'] = df['Product_Id_char'].map(category_map).fillna('Food')
    
    # 3. Harmonize sugar content categories
    if 'Product_Sugar_Content' in df.columns:
        df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace('reg', 'Regular')
        
    # 4. Calculate Store Age (Years)
    if 'Store_Establishment_Year' in df.columns:
        df['Store_Age_Years'] = 2025 - df['Store_Establishment_Year']

    # Set fallback defaults for columns to prevent pipeline mismatch
    defaults = {
        'Product_Weight': 12.66,
        'Product_Sugar_Content': 'Low Sugar',
        'Product_Allocated_Area': 0.05,
        'Product_MRP': 140.0,
        'Store_Id': 'OUT004',
        'Store_Size': 'Medium',
        'Store_Location_City_Type': 'Tier 2',
        'Store_Type': 'Supermarket Type2',
        'Product_Id_char': 'FD',
        'Product_Category': 'Food',
        'Store_Age_Years': 16
    }
    
    for col, default_val in defaults.items():
        if col not in df.columns or df[col].isnull().all():
            df[col] = default_val

    # Explicitly enforce correct training features structure and ordering
    expected_columns = [
        'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_MRP', 'Store_Id', 'Store_Size', 'Store_Location_City_Type',
        'Store_Type', 'Product_Id_char', 'Product_Category', 'Store_Age_Years'
    ]
    return df[expected_columns]

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System - Preprocessed Prediction API"

@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        data = request.get_json()
        raw_df = pd.DataFrame([data])
        processed_df = preprocess_raw_dataframe(raw_df)
        prediction = model.predict(processed_df).tolist()[0]
        return jsonify({'Sales': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 400

@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        file = request.files['file']
        raw_df = pd.read_csv(file)
        processed_df = preprocess_raw_dataframe(raw_df)
        predictions = model.predict(processed_df).tolist()
        output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}
        return jsonify(output_dict)
    except Exception as e:
        return jsonify({'error': f"Batch prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    superkart_api.run(debug=True)
