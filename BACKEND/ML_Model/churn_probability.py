import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import joblib as jb
import pandas as pd
import matplotlib.pyplot as plt
from BACKEND.ML_Model.Customer_segmentation import most_common
from BACKEND.AI_data_analyst.data_processing import get_dummies
CHURN_MODEL_PATH = os.getenv("CHURN_MODEL_PATH")
CHURN_SCALER_PATH = os.getenv("CHURN_SCALER_PATH")
model_churn = jb.load(CHURN_MODEL_PATH)
scaler_churn  = jb.load(CHURN_SCALER_PATH)
#____________________PROCESS DATA FUNCTION______________________________________
def process_data_for_churn(df_input):
    df_input.columns = df_input.columns.map(str.strip)
    cols_to_drop = {"Returns", "Age", "Total Purchase Amount", "Churn"}
    df_input.drop(columns=[col for col in cols_to_drop if col in df_input.columns], inplace=True)    
    df_input.dropna(inplace=True)
    if 'Price' not in df_input.columns:
        df_input['Price'] = df_input['Product Price']
    else:
        print("Price column already exists, skipping.") 
    df_input['TotalSpent'] = df_input['Quantity'] * df_input['Price']
    df_features = df_input.groupby("customer_id", as_index=False, sort=False).agg(
        LastPurchaseDate = ("Purchase Date","max"),
        Favoured_Product_Categories = ("Product Category", lambda x: most_common(list(x))),
        Frequency = ("Purchase Date", "count"),
        TotalSpent = ("TotalSpent", "sum"),
        Favoured_Payment_Methods = ("Payment Method", lambda x: most_common(list(x))),
        Customer_Name = ("Customer Name", "first"),
        Customer_Label = ("Customer_Labels", "first"),
    )
    df_features = df_features.drop_duplicates(subset=['Customer_Name'], keep='first')
    df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
    df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
    df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
    max_LastBuyingDate = df_features["LastPurchaseDate"].max()
    df_features['Recency'] = (max_LastBuyingDate - df_features['LastPurchaseDate']).dt.days
    df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
    df_features['Avg_Spend_Per_Purchase'] = df_features['TotalSpent']/df_features['Frequency'].replace(0,1)
    df_features['Purchase_Consistency'] = df_features['Recency'] / df_features['Frequency'].replace(0, 1)
    df_features.drop(columns=["LastPurchaseDate"],axis=1,inplace=True)
    return df_features

#___________________MODEL FUNCTION______________________________________
def encode_churn(df_features: pd.DataFrame):
    df_copy = df_features.copy()
    df_copy.drop(columns=["customer_id","Customer_Name"],axis=1,inplace=True)
    df_features_encode = get_dummies(df_copy)
    return df_features_encode

def churn_prediction(df_input:pd.DataFrame):
    df_features = process_data_for_churn(df_input)
    df_features_encode = encode_churn(df_features)
    X = scaler_churn.fit_transform(df_features_encode)
    y_pred = model_churn.predict_proba(X)[:, 1]
    df_features['Churn_Probability'] = y_pred
    return df_features

def visualize_customer_churn(df: pd.DataFrame):
    viz_tabs= st.tabs(["Customer Churn Distribution","Top 10 Customers"])
    
    with viz_tabs[0]:
        fig_dist = plt.figure(figsize=(10, 6))
        plt.hist(df['Churn_Probability'], bins=20, color='#3498db', edgecolor='black')
        plt.title('Distribution of Churn Probability')
        plt.xlabel('Churn Probability')
        plt.ylabel('Number of Customers')
        plt.axvline(df['Churn_Probability'].mean(), color='green', linestyle='dashed', label=f'Mean: {df["Churn_Probability"].mean():.2f}')
        plt.legend()
        
        st.pyplot(fig_dist)
    with viz_tabs[1]:
        top_10_high = df.nlargest(10, 'Churn_Probability')[['Customer_Name', 'Churn_Probability']]
        top_10_low = df.nsmallest(10, 'Churn_Probability')[['Customer_Name', 'Churn_Probability']]
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        bars_high = plt.barh(top_10_high['Customer_Name'], 
                            top_10_high['Churn_Probability'],
                            color='#e74c3c')
        plt.title('Highest Churn Risk Customers')
        plt.xlabel('Churn Probability')
        for bar in bars_high:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2%}',
                    ha='left', va='center', fontweight='bold')
        plt.subplot(2, 1, 2)
        bars_low = plt.barh(top_10_low['Customer_Name'],
                        top_10_low['Churn_Probability'],
                        color='#2ecc71')
        plt.title('Lowest Churn Risk Customers')
        plt.xlabel('Churn Probability')
        for bar in bars_low:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2%}',
                    ha='left', va='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)