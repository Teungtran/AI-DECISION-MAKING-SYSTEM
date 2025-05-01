import joblib as jb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from datetime import datetime as dt
from calendar import month_abbr

import streamlit as st 
XGB_MODEL = os.getenv("XGB_MODEL")
XGB_SCALER = os.getenv("XGB_SCALER")
#_______Load model____________
best_model  =  jb.load(XGB_MODEL)
scaler = jb.load(XGB_SCALER)
#______________Feature Engineering_______________
def time_features(df):
    df = df.copy()
    df = df.rename(columns={"PurchaseDate": "Date", 'TotalRevenue': "Profit"})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    target_map = df['Profit'].to_dict()
    df['lag_1week'] = (df.index - pd.Timedelta('7 days')).map(target_map)
    df['lag_2weeks'] = (df.index - pd.Timedelta('14 days')).map(target_map)
    df['lag_1month'] = (df.index - pd.Timedelta('30 days')).map(target_map)
    
    # Medium-term lags
    df['lag_quarter'] = (df.index - pd.Timedelta('91 days')).map(target_map)
    df['lag_halfyear'] = (df.index - pd.Timedelta('182 days')).map(target_map)
    
    # Yearly lags
    df['lag_1year'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag_2years'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag_3years'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    
    # Same month previous years
    df['lag_1year_exact'] = df.index.map(lambda x: 
        target_map.get(x - pd.DateOffset(years=1), None))
    df['lag_2years_exact'] = df.index.map(lambda x: 
        target_map.get(x - pd.DateOffset(years=2), None))
    return df

#_______________MODEL___________________________________

def forecasting(df, future: int):
    future = max(1, int(future))  
    df_future = df.copy()
    feature_cols = [col for col in df.columns if col not in ['Profit', 'month', 'weekofyear', 'dayofyear']]
    X_train = df[feature_cols]
    y_train = df['Profit']
    best_model.fit(X_train, y_train)
    last_record = X_train.iloc[-1:].values  
    forecast = []
    for _ in range(future):
        prediction = best_model.predict(last_record).reshape(-1, 1)
        prediction_value = float(prediction[0][0]) 
        last_record = np.hstack([last_record[:, 1:], np.array([[prediction_value]])])
        forecast.append(prediction_value)
    future_dates = pd.date_range(
        start=df_future.index.max() + pd.Timedelta(days=1),
        periods=future,
        freq='D'
    )
    future_predictions = pd.DataFrame({'Future': forecast}, index=future_dates)
    df_month = df.resample('M')['Profit'].mean().to_frame()
    df_future_month = future_predictions.resample('M').mean()
    df_future_result = pd.concat([df_month, df_future_month], axis=1)
    df_future_result['Profit'] = df_future_result['Profit'].fillna(df_future_result['Future'])
    df_final = df_future_result[['Profit']].dropna()
    return df_final, df_month.index.max(), df_final.index.max()
def visualize_report(df):
    df["Profit"] = df["TotalRevenue"]
    # Ensure the 'Date' column is in datetime format
    if 'PurchaseDate' not in df.columns:
        st.error("The 'PurchaseDate' column is missing in the dataset.")
        return
    
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    
    # Extract month, year, and day of the week from the Date column
    df['month'] = df['PurchaseDate'].dt.month
    df['year'] = df['PurchaseDate'].dt.year
    df['dayofweek'] = df['PurchaseDate'].dt.dayofweek
    
    # Create the figure for the report
    fig_report, ax = plt.subplots(3, 1, figsize=(15, 12))  
    
    # Average profit by month
    monthly_avg = df.groupby('month')['Profit'].mean()
    ax[0].plot(range(1, 13), monthly_avg.values, marker='o', linewidth=2)
    ax[0].set_xticks(range(1, 13))
    ax[0].set_xticklabels([month_abbr[i] for i in range(1, 13)])
    ax[0].set_title('Average Profit by Month')
    ax[0].set_xlabel('Month')
    ax[0].set_ylabel('Average Profit')
    
    # Average profit by day of the week
    dow_avg = df.groupby('dayofweek')['Profit'].mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ax[1].bar(days, dow_avg.values)
    ax[1].set_xticks(range(len(days)))  
    ax[1].set_title('Average Profit by Day of Week')
    ax[1].set_ylabel('Average Profit')
    
    # Yearly trend
    yearly_avg = df.groupby('year')['Profit'].mean()
    ax[2].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
    ax[2].set_title('Yearly Trend')
    ax[2].set_xlabel('Year')
    ax[2].set_ylabel('Average Profit')
    
    # Layout adjustments
    plt.tight_layout()  
    st.pyplot(fig_report)
def visualize_forecast(df, split_date=None):
    fig_forecast=plt.figure(figsize=(12, 6))
    if not split_date:
        split_date = df.dropna().index.max()
    else:
        split_date = pd.to_datetime(split_date)     
    plt.plot(df.loc[:split_date].index, df.loc[:split_date]["Profit"], label="Actual Profit", color="blue")
    plt.plot(df.loc[split_date:].index, df.loc[split_date:]["Profit"],linestyle='dashed', label="Predicted Profit", color="red")
    plt.legend()
    plt.tight_layout()  
    st.pyplot(fig_forecast)        
  
