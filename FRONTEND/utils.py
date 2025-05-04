import streamlit as st
import requests
import uuid
import os
import time
from BACKEND.AI_data_analyst.data_processing import import_data

# API endpoints
BASE_URL = "http://localhost:8888"
FORECAST_API = f"{BASE_URL}/forecast/"
CHATBOT_API = f"{BASE_URL}/chatbot/"
CHURN_API = f"{BASE_URL}/churn/"
SEGMENT_API = f"{BASE_URL}/segment/"
SENTIMENT_BATCH_API = f"{BASE_URL}/sentiment/batch/"
SENTIMENT_API = f"{BASE_URL}/sentiment/"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def configure_page(page_title, page_icon, layout="wide", initial_sidebar_state="expanded"):
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state
    )
    
    # Add custom CSS for improved UI
    st.markdown("""
    <style>
        .stButton > button {
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .feature-card {
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #f0f2f6;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 20px;
        }
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-align: center;
        }
        /* Custom styling for tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: nowrap;
            padding: 0 20px;
            border-radius: 8px 8px 0 0;
        }
        /* Custom styling for file uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #4285f4;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def feature_button(icon, label, description, key, target_page):
    """Create a styled feature button with icon, label and description"""
    with st.container():
        st.markdown(
            f"""
            <div class="feature-card">
                <h3>{icon} {label}</h3>
                <p>{description}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        if st.button(f"Explore {label}", key=key):
            st.session_state["page"] = target_page
            st.rerun()




def process_uploaded_file(uploaded_file, key_prefix=""):
    """Process uploaded file with caching"""
    if uploaded_file is not None:
        # Cache the dataframe to avoid reloading
        file_key = f"{key_prefix}_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            df = import_data(uploaded_file)
            if df is not None:
                st.session_state[file_key] = df
                st.success(f"Successfully loaded {uploaded_file.name}")
                return df
            return None
        return st.session_state[file_key]
    return None


def generate_uuid():
    """Generate a unique ID"""
    return str(uuid.uuid4())

def display_status_message(message_type, message):
    """Display status message with appropriate styling"""
    if message_type == "success":
        st.success(message)
    elif message_type == "info":
        st.info(message)
    elif message_type == "warning":
        st.warning(message)
    elif message_type == "error":
        st.error(message)
    else:
        st.write(message)