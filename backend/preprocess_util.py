import numpy as np
import pandas as pd

def preprocess_raw_data_pipeline(df_raw):
    """Transformer to dynamically clean raw input DataFrames inside the pipeline"""
    df = df_raw.copy()

    # 1. Harmonize sugar content
    if 'Product_Sugar_Content' in df.columns:
        df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace('reg', 'Regular')

    # 2. Extract Product_Id characters
    if 'Product_Id' in df.columns:
        df['Product_Id_char'] = df['Product_Id'].astype(str).str[:2]
    else:
        df['Product_Id_char'] = 'FD'

    # 3. Create mapped Product_Category
    category_map = {'FD': 'Food', 'DR': 'Drink', 'NC': 'Non-Consumable'}
    df['Product_Category'] = df['Product_Id_char'].map(category_map).fillna('Food')

    # 4. Calculate Store Age
    if 'Store_Establishment_Year' in df.columns:
        df['Store_Age_Years'] = 2025 - df['Store_Establishment_Year']
    else:
        df['Store_Age_Years'] = 16

    # Explicitly enforce correct training features structure
    expected_columns = [
        'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_MRP', 'Store_Id', 'Store_Size', 'Store_Location_City_Type',
        'Store_Type', 'Product_Id_char', 'Product_Category', 'Store_Age_Years'
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[expected_columns]
