import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go
import json
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
                # Load and prep model
        model = load_model(model_type)
        if model is None:
            return None, "Failed to load model"
        
        # Get prep path based on model type
        prep_path = get_prep_path(model_type)
        
        # Choose correct prediction function based on model type
        if model_type == 'wheat':
            result_df, error = predict_yields(result, model, prep_path)
        elif model_type == 'other_crops':
            result_df, error = predict_yields_others(result, model, prep_path)
        else:
            st.error("Invalid model type selected")
            return
            
        if error:
            st.error(error)
            return
        # Display the results
        st.subheader('Yield Prediction Results')
        st.dataframe(result_df)
        
        # Plot the boxplot
        st.subheader('Distribution of Yield Predictions Across Scenarios')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=result_df[['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 'Scenario Moderate Bad']], ax=ax)

        # Add mean values on top of each box
        for i, col in enumerate(['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 'Scenario Moderate Bad']):
            mean_val = result_df[col].mean()
            ax.text(i, mean_val, f'{mean_val:.1f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.title('Distribution of Yield Predictions Across Scenarios')
        plt.ylabel('Predicted Yield')
        st.pyplot(fig)

        # Choropleth Map
        st.subheader("Expected Yield Map")        
        try:

            geojson_filepath = os.path.join(current_dir,f'All Fields Polygons.geojson')
            
            if not os.path.exists(geojson_filepath):
                st.error("Field boundaries data file not found.")
                return
                
            with open(geojson_filepath, 'r') as f:
                geojson_data = json.load(f)
                
            selected_divisions = st.multiselect(
                "Filter by Подразделение:",
                options=sorted(result_df['Подразделение'].unique()),
                default=sorted(result_df['Подразделение'].unique())
            )

            # Filter the data
            map_data = result_df[result_df['Подразделение'].isin(selected_divisions)].copy()
            map_data = map_data[['Подразделение', 'Field_ID', 'Expected Yield']]
            
            fig_map = px.choropleth_mapbox(
                map_data, 
                geojson=geojson_data, 
                locations='Field_ID',
                featureidkey="properties.Field_ID",
                color='Expected Yield',
                color_continuous_scale="RdYlGn",
                range_color=(map_data['Expected Yield'].min(), map_data['Expected Yield'].max()),
                mapbox_style="carto-positron",
                zoom=8,
                center={"lat": 53.95, "lon": 63.48},
                opacity=0.7,
                labels={'Expected Yield': 'Expected Yield'}
            )
            
            fig_map.update_layout(
                margin={"r":0,"t":30,"l":0,"b":0},
                height=600,
                title=dict(text='Expected Yield by Field', x=0.5)
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")


if __name__ == "__main__":
    main()