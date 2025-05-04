import streamlit as st
import os
import joblib as jb
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
CLUSTER_MODEL = os.getenv("CLUSTER_MODEL")
SCALER_CLUSTER = os.getenv("SCALER_CLUSTER")
cluster_model = jb.load(CLUSTER_MODEL)
scaler_cluster = jb.load(SCALER_CLUSTER)

def most_common(lst):
    counts = Counter(lst)
    if not counts:
        return None 
    return counts.most_common(1)[0][0]
def process_data_for_model(df):
    df_model = df.copy()
    df_model.columns = df_model.columns.map(str.strip)
    df_model.dropna(inplace=True)
    if 'Price' not in df_model.columns:
        df_model['Price'] = df_model['Product Price']
    else:
        print("Price column already exists, skipping.")    
    df_model['TotalSpent'] = df_model['Quantity'] * df_model['Price'] 
    df_model.drop(columns=["Price", 'Quantity'], inplace=True)
    df_features = df_model.groupby("Customer ID", as_index=False, sort=False).agg(
        Customer_Name = ("Customer Name",'first'),
        TotalSpent=("TotalSpent", "sum"),
        Frequency=("Purchase Date", "count"),
        LastPurchaseDate=("Purchase Date", "max"),
    )
    df_features['TotalSpentFormatted'] = df_features['TotalSpent'].apply(lambda x: f"${x:,.2f}")
    df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
    max_LastBuyingDate = df_features["LastPurchaseDate"].max()
    df_features['Recency'] = (max_LastBuyingDate - df_features['LastPurchaseDate']).dt.days
    df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
    df_features = df_features.drop_duplicates(subset=['Customer_Name'], keep='first')
    return df_features

def make_cluster(df_features: pd.DataFrame):
    X = df_features[["TotalSpent", "Frequency", "Recency"]]
    scaled_X = scaler_cluster.transform(X)  
    df_features['Cluster'] = cluster_model.predict(scaled_X) 
    cluster_labels = {
        0: "VIP Customers",
        1: "Lapsed Customers",
        2: "Regular Customers",
    }
    df_features['Customer_Labels'] = df_features["Cluster"].map(cluster_labels)
    return df_features

def visualize_customer_segments(df_cluster: pd.DataFrame):
    viz_tabs = st.tabs(["Customer Segments Distribution", "Customer Segments Details"])
    with viz_tabs[0]:
        fig_dist = plt.figure(figsize=(10, 6))
        segment_counts = df_cluster['Cluster'].value_counts()
        colors = {0: '#3498db', 1: '#e74c3c', 2: '#2ecc71'}  # VIP, Lapsed, Regular
        bars = plt.bar(
            segment_counts.index,
            segment_counts.values,
            color=[colors[i] for i in segment_counts.index]
        )
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        
        plt.title('Customer Segment Distribution')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.xticks([0, 1, 2], ['VIP', 'Lapsed', 'Regular'])
        
        st.pyplot(fig_dist)

    with viz_tabs[1]:
        cluster_averages = df_cluster.groupby('Cluster').agg({
        'TotalSpent': 'mean',
        'Recency': 'mean',
        'Frequency':'median'}).reset_index()
        colors = {0: '#3498db', 1: '#e74c3c', 2: '#2ecc71'} # VIP, Lapsed, Regular
        bar_width = 0.2
        x_labels = ['TotalSpent','Recency','Frequency']
        clusters = ['0', '1', '2']

        x_positions = range(len(x_labels))
        fig_avg = plt.figure(figsize=(14, 8))
        for idx, cluster in enumerate(clusters):
            total_spent = cluster_averages.loc[cluster_averages['Cluster'] == idx, 'TotalSpent'].values[0]
            recency = cluster_averages.loc[cluster_averages['Cluster'] == idx, 'Recency'].values[0]
            frequency = cluster_averages.loc[cluster_averages['Cluster'] == idx, 'Frequency'].values[0]
            bars = plt.bar(
                [x + bar_width * idx for x in x_positions],  
                [total_spent,recency,frequency],          
                width=bar_width,
                color=colors[idx],
                label=cluster
            )
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:,.2f}',
                    ha='center',
                    va='bottom'
                )
        plt.title('Average TotalSpent and Recency by Cluster')
        plt.xlabel('Metrics')
        plt.ylabel('Average Value')
        plt.xticks([x + bar_width for x in x_positions], x_labels)  
        plt.legend(title='Cluster', labels=['VIP', 'Lapsed', 'Regular'])
        st.pyplot(fig_avg)








