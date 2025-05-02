import pandas as pd
import json
import chardet
import csv
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, HTTPException
from ML_Model.Profit_Forecasting import time_features, forecasting
from ML_Model.Customer_segmentation import process_data_for_model, make_cluster
from ML_Model.churn_probability import churn_prediction
from ML_Model.sentiment_function import analyze_customer_reviews, analyze_single_text

def df_to_records(df):
    return json.loads(df.to_json(orient='records', date_format='iso'))

async def import_data(uploaded_file):
    try:
        df = None
        
        if uploaded_file is not None:
            filename = uploaded_file.filename.lower()
            content = await uploaded_file.read()
            
            if filename.endswith(('.csv', '.txt')):
                # Use chardet to detect encoding
                result = chardet.detect(content)
                charenc = result['encoding'] or 'utf-8' 
                
                try:
                    sample = content.decode(charenc, errors='replace')[:4096]
                    dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
                    delimiter = dialect.delimiter
                except:
                    # Fallback to comma if sniffer fails
                    delimiter = ','
                

                df = pd.read_csv(BytesIO(content), encoding=charenc, delimiter=delimiter, 
                                low_memory=False, on_bad_lines='skip')
                
            # Handle Excel files
            elif filename.endswith(('.xlsx', '.xls')):
                if filename.endswith('.xlsx'):
                    engine = 'openpyxl'
                else:
                    engine = 'xlrd'
                
                df = pd.read_excel(BytesIO(content), engine=engine)
                df = convert_dates(df) 
            
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
            
            if df is None:
                raise ValueError("Unable to read file. Please check the file format.")
            if df.empty:
                raise ValueError("No data found in the file. Please check the file content.")
                
            return df
            
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

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

class ForecastController:
    @staticmethod
    async def generate_forecast(file: Optional[UploadFile] = None, future_periods: int = 12):
        try:
            if file:
                df =  await import_data(file)
            else:
                raise HTTPException(status_code=400, detail="Either file or data must be provided")

            # Process data for forecasting
            df = time_features(df)

            # Generate forecast
            forecast_df, start_date, end_date = forecasting(df, future_periods)

            return {
                "results": df_to_records(forecast_df.reset_index()),
                "start_date": start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date),
                "end_date": end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class SegmentationController:
    @staticmethod
    async def segment_customers(file: Optional[UploadFile] = None):
        try:
            if file:
                df =  await import_data(file)
            else:
                raise HTTPException(status_code=400, detail="Either file or data must be provided")
            
            df_features = process_data_for_model(df)
            
            df_clusters = make_cluster(df_features)
            
            
            return {
                "results": df_to_records(df_clusters.reset_index())
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class ChurnController:
    @staticmethod
    async def predict_churn(file: Optional[UploadFile] = None):
        try:
            # Get data from file or JSON
            if file:
                df =  await import_data(file)
            else:
                raise HTTPException(status_code=400, detail="Either file or data must be provided")
            
            # Predict churn
            df_churn = churn_prediction(df)
            
            # Return only churn predictions
            return {
                "results": df_to_records(df_churn.reset_index())
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class SentimentController:
    @staticmethod
    async def analyze_text(text: str):
        try:
            sentiment, rating = analyze_single_text(text)
            
            return {
                "sentiment": sentiment,
                "rating": float(rating[0])
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    async def analyze_batch(file: Optional[UploadFile] = None):
        try:
            if file:
                df =  await import_data(file)
            else:
                raise HTTPException(status_code=400, detail="File must be provided")

            df_sentiment = analyze_customer_reviews(df)
            return {
                "results": df_to_records(df_sentiment.reset_index())
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
