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
import importlib
import long_term_utils
from long_term_utils import (process_data_wheat, process_data_other,
    check_csv_format, predict_yields, map_crop_name
)
importlib.reload(long_term_utils)
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
            id_columns, no_outlier_df = process_data_wheat(result)
            result_df, error = predict_yields(id_columns, no_outlier_df, model, prep_path, model_type)
        elif model_type == 'other_crops':
            id_columns, no_outlier_df = process_data_other(result)
            result_df, error = predict_yields(id_columns, no_outlier_df, model, prep_path, model_type)
        else:
            st.error("Invalid model type selected")
            return
            
        if error:
            st.error(error)
            return
        # # Display the results
        # st.subheader('Yield Prediction Results')
        # st.dataframe(result_df)

        # st.subheader('Yield Prediction By Farm')
        # farm_results = result_df.groupby('Подразделение').agg({
        #     'Scenario Bad': 'mean',
        #     'Scenario Good': 'mean', 
        #     'Scenario Moderate Good': 'mean',
        #     'Scenario Moderate Bad': 'mean',
        #     'Expected Yield': 'mean'
        # }).reset_index()
        # st.dataframe(farm_results)

        # st.subheader('Summary Statistics')
        # summary_results = result_df[['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 'Scenario Moderate Bad', 'Expected Yield']].mean().to_frame().T
        # st.dataframe(summary_results)

        
        # # Plot the boxplot
        # st.subheader('Distribution of Yield Predictions Across Scenarios')
        # plt.figure(figsize=(12, 6))
        # sns.violinplot(data=result_df[['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 'Scenario Moderate Bad']], 
        #             inner='box',
        #             alpha=0.7,
        #             inner_kws={'color': 'white'})

        # for i, col in enumerate(['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 'Scenario Moderate Bad']):
        #     mean_val = result_df[col].mean()
        #     plt.text(i, mean_val, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=12)
        # plt.xticks(rotation=45)
        # plt.title('Distribution of Yield Predictions Across Scenarios')
        # plt.ylabel('Predicted Yield')
        # plt.show()
        # st.pyplot(plt)

        # # Choropleth Map
        # st.subheader("Expected Yield Map")        
        # try:

        #     geojson_filepath = os.path.join(current_dir,f'All Fields Polygons.geojson')
            
        #     if not os.path.exists(geojson_filepath):
        #         st.error("Field boundaries data file not found.")
        #         return
                
        #     with open(geojson_filepath, 'r') as f:
        #         geojson_data = json.load(f)
                
        #     selected_divisions = st.multiselect(
        #         "Filter by Подразделение:",
        #         options=sorted(result_df['Подразделение'].unique()),
        #         default=sorted(result_df['Подразделение'].unique())
        #     )

        #     # Filter the data
        #     map_data = result_df[result_df['Подразделение'].isin(selected_divisions)].copy()
        #     map_data = map_data[['Подразделение', 'Field_ID', 'Expected Yield']]
            
        #     fig_map = px.choropleth_mapbox(
        #         map_data, 
        #         geojson=geojson_data, 
        #         locations='Field_ID',
        #         featureidkey="properties.Field_ID",
        #         color='Expected Yield',
        #         color_continuous_scale="RdYlGn",
        #         range_color=(map_data['Expected Yield'].min(), map_data['Expected Yield'].max()),
        #         mapbox_style="carto-positron",
        #         zoom=8,
        #         center={"lat": 53.95, "lon": 63.48},
        #         opacity=0.7,
        #         labels={'Expected Yield': 'Expected Yield'}
        #     )
            
        #     fig_map.update_layout(
        #         margin={"r":0,"t":30,"l":0,"b":0},
        #         height=600,
        #         title=dict(text='Expected Yield by Field', x=0.5)
        #     )
            
        #     st.plotly_chart(fig_map, use_container_width=True)
            
        # except Exception as e:
        #     st.error(f"Error creating map: {str(e)}")
        
        def display_basic_stats(result_df: pd.DataFrame, by_crop: bool = False):
            """Display basic statistics for yield predictions"""
            st.subheader('Yield Prediction Results')
            st.dataframe(result_df)

            groupby_cols = ['Подразделение']
            if by_crop:
                groupby_cols.append('Культура')
                result_df = map_crop_name(result_df)

            st.subheader('Yield Prediction By Farm' + (' and Crop' if by_crop else ''))
            farm_results = result_df.groupby(groupby_cols).agg({
                'Scenario Bad': 'mean',
                'Scenario Good': 'mean', 
                'Scenario Moderate Good': 'mean',
                'Scenario Moderate Bad': 'mean',
                'Expected Yield': 'mean'
            }).reset_index()
            st.dataframe(farm_results)

            if by_crop:
                st.subheader('Summary Statistics by Crop')
                crop_summary = result_df.groupby('Культура')[
                    ['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 
                    'Scenario Moderate Bad', 'Expected Yield']
                ].mean().reset_index()
                st.dataframe(crop_summary)
            else:
                st.subheader('Overall Summary Statistics')
                summary_results = result_df[
                    ['Scenario Bad', 'Scenario Good', 'Scenario Moderate Good', 
                    'Scenario Moderate Bad', 'Expected Yield']
                ].mean().to_frame().T
                st.dataframe(summary_results)

        def plot_yield_distribution(result_df: pd.DataFrame, by_crop: bool = False):
            """Plot yield distribution across scenarios"""
            if by_crop:
                crops = result_df['Культура'].unique()
                for crop in crops:
                    st.subheader(f'Distribution of Yield Predictions - {crop}')
                    crop_df = result_df[result_df['Культура'] == crop]
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.violinplot(
                        data=crop_df[['Scenario Bad', 'Scenario Good', 
                                    'Scenario Moderate Good', 'Scenario Moderate Bad']], 
                        inner='box',
                        alpha=0.7,
                        inner_kws={'color': 'white'}
                    )
                    
                    for i, col in enumerate(['Scenario Bad', 'Scenario Good', 
                                        'Scenario Moderate Good', 'Scenario Moderate Bad']):
                        mean_val = crop_df[col].mean()
                        plt.text(i, mean_val, f'{mean_val:.1f}', 
                                ha='center', va='bottom', fontsize=12)
                    
                    plt.xticks(rotation=45)
                    plt.title(f'Distribution of Yield Predictions Across Scenarios - {crop}')
                    plt.ylabel('Predicted Yield')
                    st.pyplot(fig)
                    plt.close()
            else:
                st.subheader('Distribution of Yield Predictions Across Scenarios')
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.violinplot(
                    data=result_df[['Scenario Bad', 'Scenario Good', 
                                'Scenario Moderate Good', 'Scenario Moderate Bad']], 
                    inner='box',
                    alpha=0.7,
                    inner_kws={'color': 'white'}
                )
                
                for i, col in enumerate(['Scenario Bad', 'Scenario Good', 
                                    'Scenario Moderate Good', 'Scenario Moderate Bad']):
                    mean_val = result_df[col].mean()
                    plt.text(i, mean_val, f'{mean_val:.1f}', 
                            ha='center', va='bottom', fontsize=12)
                
                plt.xticks(rotation=45)
                plt.title('Distribution of Yield Predictions Across Scenarios')
                plt.ylabel('Predicted Yield')
                st.pyplot(fig)
                plt.close()

        def display_yield_map(result_df: pd.DataFrame, by_crop: bool = False):
            """Display choropleth map of expected yields"""
            try:
                geojson_filepath = os.path.join(current_dir, 'All Fields Polygons.geojson')
                if not os.path.exists(geojson_filepath):
                    st.error("Field boundaries data file not found.")
                    return
                    
                with open(geojson_filepath, 'r') as f:
                    geojson_data = json.load(f)

                # Create filters
                filters = {}
                filters['Подразделение'] = st.multiselect(
                    "Filter by Подразделение:",
                    options=sorted(result_df['Подразделение'].unique()),
                    default=sorted(result_df['Подразделение'].unique())
                )

                if by_crop:
                    filters['Культура'] = st.multiselect(
                        "Filter by Crop:",
                        options=sorted(result_df['Культура'].unique()),
                        default=sorted(result_df['Культура'].unique())
                    )

                # Apply filters
                map_data = result_df.copy()
                for col, selected_values in filters.items():
                    map_data = map_data[map_data[col].isin(selected_values)]

                # Prepare map data
                display_cols = ['Подразделение', 'Field_ID', 'Expected Yield']
                if by_crop:
                    display_cols.append('Культура')
                map_data = map_data[display_cols]

                hover_data = {'Expected Yield': ':.2f'}
                if by_crop:
                    hover_data['Культура'] = True

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
                    labels={'Expected Yield': 'Expected Yield'},
                    hover_data=hover_data
                )
                
                fig_map.update_layout(
                    margin={"r":0,"t":30,"l":0,"b":0},
                    height=600,
                    title=dict(text='Expected Yield by Field', x=0.5)
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")

        def display_results(result_df: pd.DataFrame, model_type: str):
            """Main function to display all results"""
            by_crop = model_type == 'other_crops'
            
            display_basic_stats(result_df, by_crop)
            plot_yield_distribution(result_df, by_crop)
            
            st.subheader("Expected Yield Map")
            display_yield_map(result_df, by_crop)
        display_results(result_df, model_type)

if __name__ == "__main__":
    main()