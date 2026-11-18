
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

st.title("SuperKart System - Multi-Format Batch Ingestion UI")
st.write(
    "Enter product details below for a single prediction, or upload any raw CSV batch file for instant processing."
)

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Single Prediction
if st.button("Predict", type='primary'):
    response = requests.post(
        f"{BACKEND_URL}/v1/predict",
        json=product_data
    )

    if response.status_code == 200:
        result = response.json()
        predicted_sales = result["Sales"]
        st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:.2f}")
    else:
        st.error(f"API Error: {response.text}")

# Batch Prediction
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload a raw CSV file (supports 10-column or raw inputs)", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict for Batch", type='primary'):
        # Let's read it to verify formatting locally if needed
        df_raw = pd.read_csv(uploaded_file)
        
        # Reset buffer
        uploaded_file.seek(0)
        
        # Send the file directly to our backend endpoint
        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            results = response.json()
            st.success("Predictions completed successfully!")
            
            # Map predictions back to the original rows
            pred_series = pd.Series(results).astype(float)
            df_raw['Predicted_Sales'] = pred_series.values
            
            st.dataframe(df_raw, use_container_width=True)
        else:
            st.error(f"Unable to complete prediction batch: {response.text}")
