import pandas as pd
import chardet
import csv
from io import BytesIO
import streamlit as st
def import_data(uploaded_file):
    """
    Import data from uploaded Excel or CSV files with improved delimiter detection
    and robust error handling.
    """
    try:
        df = None
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Handle Excel files
        if file_extension in ['xlsx', 'xls']:
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            except ImportError:
                # Fallback if openpyxl not available
                df = pd.read_excel(uploaded_file)
        
        # Handle CSV files
        elif file_extension in ['csv', 'txt']:
            # Try common delimiters in order of likelihood
            delimiters = [',', ';', '\t', '|']
            for delimiter in delimiters:
                try:
                    # Try to read with different encodings
                    for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(uploaded_file, 
                                           sep=delimiter, 
                                           encoding=encoding,
                                           engine='python',  # More flexible engine
                                           on_bad_lines='warn')  # Continue on bad lines
                            if not df.empty and len(df.columns) > 1:  # Ensure we actually got data
                                break  # Successfully loaded
                        except Exception:
                            continue
                    if df is not None and not df.empty and len(df.columns) > 1:
                        break  # Successfully loaded
                except Exception:
                    continue
        
        # Check if data was successfully loaded
        if df is None or df.empty or len(df.columns) <= 1:
            st.error("Unable to read file or data has incorrect format. Please check your file.")
            return None
            
        # Clean column names
        df.columns = df.columns.str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"Error importing data: {str(e)}")
        return None

def detect_outliers(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    outliers_summary = {}
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_count = outliers.shape[0]
        if outliers_count > 0:
            outliers_summary[col] = outliers_count
        else:
            outliers_summary[col] = 0
    
    if len(outliers_summary) > 0:
        outliers_info = "\n".join([f"{col}: {count} outliers" for col, count in outliers_summary.items()])
    else:
        outliers_info = "No outliers detected."
    
    return outliers_info
def analyze_dataset(df):
    if not isinstance(df, pd.DataFrame):
        return "Input is not a valid DataFrame."
    
    num_rows, num_columns = df.shape
    dataset_info = f"Dataset Info:\n- Number of rows: {num_rows} : - Number of columns: {num_columns}\n\n"
    
    data_types = df.dtypes
    dataset_info += "Data Types of Columns:\n" + data_types.apply(lambda x: f": {x}").to_string() + "\n\n"
    
    null_values = df.isnull().sum()
    dataset_info += "Null Values per Column:\n" + null_values.apply(lambda x: f": {x}").to_string() + "\n\n"
    
    outliers_info = detect_outliers(df)
    dataset_info += f"\nOutliers Summary:\n{outliers_info}"
    
    return dataset_info

def convert_dates(df):
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ''
            if isinstance(sample, str) and any(char in sample for char in ['-', '/', '.', ':']):
                date_columns.append(col)
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].dt.date
    return df
def get_dummies(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if df[col].isin(['yes', 'no', 'True', 'False']).any():
                df[col] = df[col].map({'yes': 1, 'True': 1, 'no': 0, 'False': 0})
            else:
                df = pd.get_dummies(df, columns=[col])
    return df