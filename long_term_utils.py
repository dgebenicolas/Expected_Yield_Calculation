import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import streamlit as st

# Add the current_dir variable
current_dir = os.path.dirname(os.path.abspath(__file__))


REQUIRED_COLUMNS = ['Подразделение', 'Поле', 'Field_ID',
'Year', 'Агрофон', 'Культура', 'Macro Total/ha',
       'Fung Total/ha', 'Pest Total/ha', 'bdod', 'phh2o', 'sand', 'silt',
       'soc', 'DOY_min'
]


COLUMN_DTYPES = {
    'Year': 'int64', 'Агрофон': 'object', 'Культура': 'object', 
    'Macro Total/ha': 'float64', 'Fung Total/ha': 'float64', 'Pest Total/ha': 'float64', 
    'bdod': 'float64', 'phh2o': 'float64', 'sand': 'float64', 'silt': 'float64', 'soc': 'float64', 
    'DOY_min': 'int64'
}

REQUIRED_COLUMNS_2 = ['Подразделение', 'Поле', 'Field_ID',
'Year', 'Агрофон', 'Культура', 'Fung Total/ha',
       'Pest Total/ha', 'bdod', 'cec', 'clay', 'phh2o', 'sand', 'silt', 'soc',

]

COLUMN_DTYPES_2 = {
    'Year': 'int64', 'Агрофон': 'object', 'Культура': 'object', 
     'Fung Total/ha': 'float64', 'Pest Total/ha': 'float64', 
    'bdod': 'float64','cec': 'float64', 'clay': 'float64', 'phh2o': 'float64', 'sand': 'float64', 'silt': 'float64', 'soc': 'float64', 

}


def setup_preprocessor(pre_process_df):
    numeric_features = list(pre_process_df.drop(['Агрофон', 'Культура'], axis=1).select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = ['Агрофон', 'Культура']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    preprocessor.fit(pre_process_df)
    return preprocessor, numeric_features, categorical_features

def check_csv_format(file):
    if not file.name.endswith('.csv'):
        return False, "File must be a CSV"
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return False, f"Error reading CSV file: {str(e)}"
    
    return True, df

def process_data_wheat(df):
    # Store IDs before processing
    id_columns = df[['Подразделение', 'Поле','Field_ID']].copy()
    
    # Drop ID columns and reorder remaining columns
    process_cols = [col for col in REQUIRED_COLUMNS if col not in ['Подразделение', 'Поле','Field_ID']]
    process_df = df[process_cols].copy()
    
    # Enforce data types
    for col, dtype in COLUMN_DTYPES.items():
        if col in process_df.columns:
            process_df[col] = process_df[col].astype(dtype)
    
    return id_columns, process_df

def process_data_other(df):
    # Store IDs before processing
    id_columns = df[['Подразделение', 'Поле','Field_ID', 'Культура']].copy()
    
    # Drop ID columns and reorder remaining columns
    process_cols = [col for col in REQUIRED_COLUMNS_2 if col not in ['Подразделение', 'Поле','Field_ID']]
    process_df = df[process_cols].copy()
    
    # Enforce data types
    for col, dtype in COLUMN_DTYPES_2.items():
        if col in process_df.columns:
            process_df[col] = process_df[col].astype(dtype)
    
    return id_columns, process_df

def map_agrofon_to_group(df):
    """
    Maps the 'Агрофон' column in the given DataFrame to predefined product groups.

    The function categorizes various agrofon types into broader groups such as 
    'Stubble', 'Fallow', and 'Deep Tillage' based on the presence of specific 
    keywords in the agrofon names. If an agrofon does not match any of the 
    predefined categories, it is labeled as 'others'.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing an 'Агрофон' column.

    Returns:
    pd.DataFrame: The modified DataFrame with the 'Агрофон' column updated to 
    reflect the mapped product groups.
    """
    mapped_df = df.copy()
    def map_product_name(product_name):
        product_name = product_name.lower()  # Convert to lower case

        if "стерня" in product_name:
            return "Stubble"
        elif "пар" in product_name:
            return "Fallow"
        elif "глубокая" in product_name or "глубокое" in product_name:
            return "Deep Tillage"
        return 'others' 
    
    mapped_df['Агрофон'] = mapped_df['Агрофон'].apply(map_product_name)
    
    return mapped_df

def rename_product_groups(df):
    """
    Maps product names in the given DataFrame to standardized product groups.
    
    The function categorizes various wheat (пшеница) varieties into their respective
    groups based on the presence of specific variety names in the product names.
    The matching is case-insensitive and handles variations in product name formats.
    If a product doesn't match any predefined variety, it is labeled as 'others'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'Product Name' column.
    
    Returns:
    pd.DataFrame: The modified DataFrame with an additional 'Product Group' column 
                 containing the mapped product groups.
    """
    mapped_df = df.copy()
    
    def map_product_name(product_name):
        # Convert to lower case for case-insensitive matching
        product_name = str(product_name).lower()
        
        # Dictionary of variety keywords and their standardized group names
        variety_groups = {
            'астана': 'Астана',
            'шортандинская': 'Шортандинская',
            'ликамеро': 'Ликамеро',
            'тобольская': 'Тобольская',
            'тризо': 'Тризо',
            'радуга': 'Радуга',
            'гранни': 'Гранни',
            'урало-сибирская': 'Урало-Сибирская',
            'уралосибирская': 'Урало-Сибирская',  # Handle variation in spelling
            'айна': 'Айна'
        }
        
        # Check for each variety keyword in the product name
        for keyword, group in variety_groups.items():
            if keyword in product_name:
                return group
                
        return 'others'
    
    mapped_df['Культура'] = mapped_df['Культура'].apply(map_product_name)
    
    return mapped_df

def map_crop_name(df):
    mapped_df = df.copy()
    def map_product_name(product_name):
        product_name = product_name.lower()  # Convert to lower case

        if product_name.startswith("лен"):
            return "Flakes"
        elif product_name.startswith("пшеница твердая"):
            return "Hard Wheat"
        elif product_name.startswith("подсолнечник"):
            return "Sunflower"
        return 'others' 
    
    mapped_df['Культура'] = mapped_df['Культура'].apply(map_product_name)
    
    return mapped_df


def predict_yields(id_columns, no_outlier_df, model, prep_path, model_type):
    """
    Main prediction function that handles data processing and yield predictions
    Args:
        df (pd.DataFrame): Input DataFrame with required features
        model_type (str): Either 'wheat' or 'other_crops'
    Returns:
        tuple: (result_df, error_message)
    """
    try:
        
        # Load climate data
        aggregated_df = pd.read_csv(os.path.join(current_dir, 'aggregated_df.csv'))
        climate_columns = [col for col in aggregated_df.columns if col != 'climate_quantile']
            
        # Process climate scenarios
        dfs = []
        for _, climate_row in aggregated_df.iterrows():
            temp_df = no_outlier_df.copy()
            temp_df[climate_columns] = climate_row[climate_columns].values
            dfs.append(temp_df)
        
        # Load and setup preprocessor
        prep_df = pd.read_csv(os.path.join(current_dir, prep_path))
        prep_df = map_agrofon_to_group(prep_df)
        if model_type == 'wheat':
            prep_df = rename_product_groups(prep_df)
        elif model_type == 'other_crops':
            prep_df = map_crop_name(prep_df)
        preprocessor, numeric_features, categorical_features = setup_preprocessor(prep_df)
        feature_names = (numeric_features + 
                        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
        
        # Generate predictions
        y_pred_list = []
        for df in dfs:
            pre_process_df = map_agrofon_to_group(df)
            if model_type == 'wheat':
                pre_process_df = rename_product_groups(pre_process_df)
            elif model_type == 'other_crops':
                pre_process_df = map_crop_name(pre_process_df)
            processed_data = preprocessor.transform(pre_process_df)
            processed_df = pd.DataFrame(processed_data, columns=feature_names)
            if 'Культура_others' in processed_df.columns:
                processed_df = processed_df.drop(columns='Культура_others')
            y_pred = model.predict(processed_df)
            y_pred_list.append(y_pred)
        
        # Create results DataFrame
        predictions_df = pd.DataFrame({
            'Scenario Bad': y_pred_list[0],
            'Scenario Good': y_pred_list[1],
            'Scenario Moderate Good': y_pred_list[3],
            'Scenario Moderate Bad': y_pred_list[2]
        })
        
        # Calculate expected yield
        predictions_df['Expected Yield'] = (
            predictions_df['Scenario Bad'] * 0.2 +
            predictions_df['Scenario Good'] * 0.2 +
            predictions_df['Scenario Moderate Good'] * 0.3 +
            predictions_df['Scenario Moderate Bad'] * 0.3
        )
        
        # Combine results
        result_df = pd.concat([id_columns, predictions_df], axis=1)
        return result_df, None
        
    except Exception as e:
        return None, f"Error in prediction pipeline: {str(e)}"


__all__ = [
    'setup_preprocessor',
    'check_csv_format',
    'process_data_wheat',
    'map_agrofon_to_group',
    'REQUIRED_COLUMNS',
    'COLUMN_DTYPES',
    'REQUIRED_COLUMNS_2',
    'COLUMN_DTYPES_2',
    'rename_product_groups',
    'process_data_other',
    'map_crop_name',
    'predict_yields',
    'predict_yields_others',
    'load_model'  # Add this line
]