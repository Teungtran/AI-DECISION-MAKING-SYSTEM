import streamlit as st
from FRONTEND.utils import generate_uuid

def init_session_state():
    """Initialize session state variables"""
    # Page navigation
    if "page" not in st.session_state:
        st.session_state["page"] = "🏠 Home"
    
    # Current feature
    if "current_feature" not in st.session_state:
        st.session_state["current_feature"] = None
    
    # Chatbot session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_uuid()
    
    # Chat history for data analyst
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Feature specific states
    if "df_churn_prediction" not in st.session_state:
        st.session_state.df_churn_prediction = None
        
    if "df_cluster" not in st.session_state:
        st.session_state.df_cluster = None
        
    if "df_sentiment_report" not in st.session_state:
        st.session_state.df_sentiment_report = None
        
    if "forecast_data" not in st.session_state:
        st.session_state.forecast_data = None
        
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    
    if "show_text_input" not in st.session_state:
        st.session_state.show_text_input = False
        
    if "show_batch_analysis" not in st.session_state:
        st.session_state.show_batch_analysis = False

def update_page(page_name):
    """Update current page in session state"""
    st.session_state["page"] = page_name
    st.session_state["current_feature"] = page_name

def clear_chat_history():
    """Clear chat history"""
    st.session_state.chat_history = []
    st.success("Chat history cleared")

def reset_feature_state(feature_name):
    """Reset feature-specific session state"""
    if feature_name == "Chatbot":
        st.session_state.messages = []
        st.session_state.session_id = generate_uuid()
    elif feature_name == "Predict Churn Probability":
        st.session_state.df_churn_prediction = None
    elif feature_name == "Customer Segmentation":
        st.session_state.df_cluster = None
    elif feature_name == "Customer Feedback Analysis":
        st.session_state.df_sentiment_report = None
        st.session_state.show_text_input = False
        st.session_state.show_batch_analysis = False
    elif feature_name == "Forecasting Profit":
        st.session_state.forecast_data = None
        st.session_state.original_df = None