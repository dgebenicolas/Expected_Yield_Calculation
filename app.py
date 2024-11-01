import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go
from long_term_utils import (
    setup_preprocessor, check_csv_format, process_data_wheat, 
    map_agrofon_to_group, REQUIRED_COLUMNS, COLUMN_DTYPES, REQUIRED_COLUMNS_2, COLUMN_DTYPES_2, rename_product_groups,
    process_data_other, map_crop_name, predict_yields, predict_yields_others
    
)
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(model_type):
    """
    Load model based on user selection
    Args:
        model_type (str): Either 'wheat' or 'other_crops'
    Returns:
        object: Loaded model or None if error
    """
    model_paths = {
        'wheat': 'long_term_lgbm.txt',
        'other_crops': 'other_crops_lgbm.txt'
    }
    
    try:
        model_path = os.path.join(current_dir, model_paths[model_type])
        model = lgb.Booster(model_file=model_path)
        return model
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        return None

def get_prep_path(model_type):
    """
    Get preprocessing data path based on model type
    """
    prep_paths = {
        'wheat': 'long_term_test.csv',
        'other_crops': 'other_crop_set_up.csv'
    }
    return prep_paths[model_type]

def main():
    st.title('Crop Yield Prediction')
    
    # Add model selection dropdown
    model_type = st.selectbox(
        "Select crop type for prediction",
        options=['wheat', 'other_crops'],
        index=0,
        help="Choose 'wheat' for wheat predictions or 'other_crops' for other crop types"
    )
    
    # File uploader for input data
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Check CSV format
        is_valid, result = check_csv_format(uploaded_file)
        if not is_valid:
            st.error(result)
            return
        
        # Get prep path based on model type
        prep_path = get_prep_path(model_type)
        
        # Choose correct prediction function based on model type
        if model_type == 'wheat':
            result_df, error = predict_yields(result, model_type, prep_path)
        else:
            result_df, error = predict_yields_others(result, model_type, prep_path)
            
        if error:
            st.error(error)
            return
        # Display the results
        st.subheader('Yield Prediction Results')
        st.dataframe(result_df)
        
        # Plot the boxplot
        st.subheader('Distribution of Yield Predictions Across Scenarios')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=predictions_df[['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 'Scenario Moderate Bad']], ax=ax)

        # Add mean values on top of each box
        for i, col in enumerate(['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 'Scenario Moderate Bad']):
            mean_val = predictions_df[col].mean()
            ax.text(i, mean_val, f'{mean_val:.1f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.title('Distribution of Yield Predictions Across Scenarios')
        plt.ylabel('Predicted Yield')
        st.pyplot(fig)

if __name__ == "__main__":
    main()