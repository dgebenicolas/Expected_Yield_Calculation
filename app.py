import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from long_term_utils import (
    setup_preprocessor, check_csv_format, process_data, 
    map_agrofon_to_group, REQUIRED_COLUMNS, COLUMN_DTYPES, rename_product_groups
)
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_model():
    try:
        model = lgb.Booster(model_file=os.path.join(current_dir, 'long_term_lgbm.txt'))
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title('Crop Yield Prediction')
    
    # File uploader for no_outlier_df
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Check CSV format
        is_valid, result = check_csv_format(uploaded_file)
        if not is_valid:
            st.error(result)
            return
        
    df = result.copy()  # result is the DataFrame if validation passed
    id_columns, no_outlier_df = process_data(df)
    # Load the aggregated climate data
    try:
        aggregated_df = pd.read_csv(os.path.join(current_dir, 'aggregated_df.csv'))
    except Exception as e:
        st.error(f"Error loading aggregated climate data: {str(e)}")
        return
    
    # Load the pre-trained LightGBM model
    model = load_model()
    if model is None:
        return

    # Get climate feature columns
    climate_columns = [col for col in aggregated_df.columns if col != 'climate_quantile']
    
    # Process each climate scenario and save separately
    dfs = []
    for idx, climate_row in aggregated_df.iterrows():
        temp_df = no_outlier_df[no_outlier_df['Year'] == 2024].copy()
        temp_df[climate_columns] = climate_row[climate_columns].values
        dfs.append(temp_df)
    
    # Preprocess the data
    prep_df = pd.read_csv(os.path.join(current_dir, 'prep_data.csv'))
    prep_df = map_agrofon_to_group(prep_df)
    prep_df = rename_product_groups(prep_df)
    preprocessor, numeric_features, categorical_features = setup_preprocessor(prep_df)
    processed_data = preprocessor.transform(prep_df)
    feature_names = (numeric_features + 
                    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
    
    # Make predictions
    y_pred_list = []
    
    for df in dfs:
        pre_process_df = map_agrofon_to_group(pre_process_df)
        pre_process_df = rename_product_groups(pre_process_df)
        processed_data = preprocessor.transform(pre_process_df)
        processed_df = pd.DataFrame(processed_data, columns=feature_names)
        y_pred = model.predict(processed_df)
        y_pred_list.append(y_pred)
    
    # Create a dataframe with the predictions
    predictions_df = pd.DataFrame({
        'Scenario Bad': y_pred_list[0],
        'Scenario Good': y_pred_list[1],
        'Scenario Moderate Good': y_pred_list[3],
        'Scenario Moderate Bad': y_pred_list[2]
    })
    
    # Calculate the expected yield
    predictions_df['Expected Yield'] = (
        predictions_df['Scenario Bad'] * 0.2 +
        predictions_df['Scenario Good'] * 0.2 +
        predictions_df['Scenario Moderate Good'] * 0.3 +
        predictions_df['Scenario Moderate Bad'] * 0.3
    )
    
    # Display the results
    st.subheader('Yield Prediction Results')
    result_df = pd.concat([id_columns, predictions_df], axis=1)
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