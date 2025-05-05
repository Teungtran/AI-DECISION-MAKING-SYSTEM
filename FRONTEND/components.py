import streamlit as st
import requests
import os
import time
import pandas as pd
import json
from BACKEND.ML_Model.Customer_segmentation import visualize_customer_segments
from BACKEND.ML_Model.sentiment_function import word_cloud, sentiment_distribution
from BACKEND.ML_Model.churn_probability import visualize_customer_churn
from BACKEND.ML_Model.Profit_Forecasting import visualize_forecast, visualize_report
from BACKEND.AI_data_analyst.chat_csv import setup_chat_history, handle_chat_interface, clear_chat_history, visualize_data
from BACKEND.AI_data_analyst.data_processing import  get_dummies, analyze_dataset
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from FRONTEND.utils import (
    process_uploaded_file, 
    CHATBOT_API,
    FORECAST_API,
    CHURN_API,
    SEGMENT_API,
    SENTIMENT_BATCH_API,
    SENTIMENT_API,
    GOOGLE_API_KEY,
    display_status_message
)

from FRONTEND.state import update_page, reset_feature_state

def back_to_home_button():
    """Render a styled back to home button"""
    if st.sidebar.button("← Back to Home", key="back_btn"):
        update_page("🏠 Home")
        
def render_chatbot_page():
    """Render the chatbot page with thinking spinner"""
    back_to_home_button()
    st.title("🛒 Cortex AI Chatbot 🤖")
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = ""
    if "thinking" not in st.session_state:
        st.session_state.thinking = False
    
    # User settings section
    with st.expander("User Settings & Chat Tools", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.session_state.session_id = st.text_input(
                "User ID (optional):", 
                value=st.session_state.session_id
            )
        
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Sample questions section
    with st.expander("💡 Try asking:", expanded=True):
        sample_questions = [
            ("📦 Amazon seller policy", "What is Amazon policy on selling platform for sellers"),
            ("👑 VIP Customer stats", "What is the number of VIP Customers labeled, and their average money spent"),
            ("📣 Teen marketing campaign", "How can I set up a marketing campaign aimed at teen customers"),
            ("📊 Today's business trends", "What is today's latest business trends")
        ]
        
        cols = st.columns(2)
        for i, (label, question) in enumerate(sample_questions):
            if cols[i % 2].button(label, key=f"sample_q_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.thinking = True
                st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Show thinking spinner if active
    if st.session_state.thinking:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                process_message(st.session_state.messages[-1]["content"])
                st.session_state.thinking = False
                st.rerun()

    # Process user input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.thinking = True
        st.rerun()

def process_message(user_input):
    """Process a message and get AI response"""
    try:
        payload = {
            "user_input": user_input, 
            "user_id": st.session_state.session_id,
            "delay": 0.005
        }
        
        accumulated_response = ""
        
        # Stream the response
        with requests.post(
            CHATBOT_API, 
            json=payload, 
            stream=True, 
            timeout=120
        ) as response:
            response.raise_for_status()
            
            # Process streaming response
            for chunk in response.iter_content(chunk_size=1):
                if chunk:
                    try:
                        text = chunk.decode('utf-8')
                        accumulated_response += text
                    except UnicodeDecodeError:
                        pass
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": accumulated_response
        })
        
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Sorry, I encountered an error: {str(e)}"
        })
def render_data_analyst_page():
    """Render the AI Data Analyst page"""
    back_to_home_button()
    st.title("📊 AI Data Analyst")
    st.markdown("**Explore your data in an interactive way!**")
    st.divider()
    
    
    # File upload with caching
    uploaded_file = st.file_uploader(
        "Upload your CSV or XLSX file", 
        type=['csv', 'xlsx'], 
        key="analyst_file"
    )
    
    df = process_uploaded_file(uploaded_file, key_prefix="analyst")
    
    if df is not None:
        tabs = st.tabs(["Data Preview", "Data Analysis", "Data Visualization", "AI Chat"])
        
        with tabs[0]:
            st.dataframe(df.head(100), use_container_width=True)
            if st.button("Convert Categorical data to Numeric", key="convert_cat"):
                with st.spinner("Converting data..."):
                    df_converted = get_dummies(df)
                    st.dataframe(df_converted.head(100), use_container_width=True)
        
        with tabs[1]:
            if st.button("Analyze Dataset", key="analyze_data"):
                with st.spinner("Analyzing dataset..."):
                    dataset_info = analyze_dataset(df)
                    st.text_area("Dataset Analysis", dataset_info, height=400)
        
        with tabs[2]:
            visualize_data(df)
        
        with tabs[3]:
            setup_chat_history()
            handle_chat_interface(df, GOOGLE_API_KEY, None)
            if st.button("Clear Chat History", key="clear_chat"):
                clear_chat_history()

def render_forecasting_page():
    """Render the forecasting profit page"""
    st.title("📈 Forecasting Profit")
    st.markdown("**🔍📈💰 Where's the money going? Let's find out! 💸💰**")
    st.divider()
    back_to_home_button()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV or XLSX file", 
        type=['csv', 'xlsx'],
        key="forecast_file"
    )
    
    if uploaded_file is not None:
        df = process_uploaded_file(uploaded_file, key_prefix="forecast")
        
        with st.expander("Data Preview"):
            st.dataframe(df.head(100), use_container_width=True)
        
        # Input parameters
        future_periods = st.number_input(
            "Number of periods to forecast (days)",
            min_value=30,
            max_value=1000,
            value=365,
            step=30,
            help="Select how many days into the future you want to forecast"
        )
        

        if st.button("🔮 Look into the Future!", key="run_forecast", use_container_width=True):
            with st.spinner("Processing data..."):
                try:
                    uploaded_file.seek(0)
                    files = {'file': uploaded_file}
                    payload = {'future_periods': future_periods}
                    
                    response = requests.post(FORECAST_API, data=payload, files=files)

                    
                    if response:
                        forecast_data = response.json()
                        st.session_state['forecast_data'] = forecast_data
                        st.session_state['original_df'] = df
                    # Display results if available
                    if 'forecast_data' in st.session_state and st.session_state['forecast_data']:
                        forecast_data = st.session_state['forecast_data']
                        forecast_df = pd.DataFrame(forecast_data['results'])
            
                    # Convert to datetime index if results contain date strings
                    if 'index' in forecast_df.columns:
                        forecast_df.set_index('index', inplace=True)
                        forecast_df.index = pd.to_datetime(forecast_df.index)
            
                    start_date = pd.to_datetime(forecast_data['start_date'])
                    end_date = forecast_data['end_date']
                    
                    with st.expander("Forecasting Preview"):
                        st.dataframe(forecast_df, use_container_width=True)
                        st.markdown(f"**Forecast Start Date:** {start_date}")
                        st.markdown(f"**Forecast End Date:** {end_date}")
                    
                    forecast_tabs = st.tabs(['Forecasting Visualization', 'Financial Report'])
                    
                    with forecast_tabs[0]:
                        try:
                            visualize_forecast(forecast_df, start_date)
                        except Exception as e:
                            display_status_message("error", f"Error in visualization: {str(e)}")
                    
                    with forecast_tabs[1]:
                        try:
                            if 'original_df' in st.session_state:
                                visualize_report(st.session_state['original_df'])
                            else:
                                display_status_message("warning", "Original data not available for reporting")
                        except Exception as e:
                            display_status_message("error", f"Error generating report: {str(e)}")
                except Exception as e:
                    display_status_message("error", f"Error in forecast: {str(e)}")

def render_churn_page():
    """Render the Churn Prediction page"""
    st.title("⚠️💸 Predict Churn Probability")
    st.markdown("🔍📉🤔 Customer slipping away? Not on our watch! 👀🚀")
    st.divider()
    back_to_home_button()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload with caching
        uploaded_file = st.file_uploader("Upload your **CSV** or **XLSX** file", type=['csv', 'xlsx'], key="churn_file")
        df = process_uploaded_file(uploaded_file)
    
    with col2:
        with st.expander("What data do I need?"):
            st.info("""
            #### Required Columns
            - `customer_id`  
            - `Purchase Date`  
            - `Product Category`  
            - `Product Price`  
            - `Quantity`  
            - `Payment Method`  
            - `Customer Name`  
            - `Customer_Labels`  
            - `Price`  

            #### Output After Prediction
            - `Favoured_Product_Categories`  
            - `Frequency`  
            - `TotalSpent`  
            - `Favoured_Payment_Methods`  
            - `Customer_Label`  
            - `Recency`  
            - `Avg_Spend_Per_Purchase`  
            - `Purchase_Consistency`  
            - `Churn_Probability`  
            """)
    
    if df is not None:
        with st.expander("📄 Data Preview", expanded=True):
            st.dataframe(df.head(100))
        predict_button = st.button("Predict Churn Probability", use_container_width=True)
        if predict_button:
            with st.spinner("Running churn prediction..."):
                try:
                    # Reset file position for reading
                    uploaded_file.seek(0)
                    files = {'file': uploaded_file}
                    response = requests.post(CHURN_API, files=files)
                    
                    if response:
                        response_data = response.json()
                    df_churn_prediction = pd.DataFrame(response_data["results"])
                    st.session_state.df_churn_prediction = df_churn_prediction
                    
                    with st.expander("📊 Churn Probability Results", expanded=True):
                        st.dataframe(df_churn_prediction.head(100))
                    
                    st.success("✅ Churn Prediction completed!")
                    
                except requests.exceptions.RequestException as e:
                    st.error("Network error")
                    st.exception(e)
                except Exception as e:
                    st.error("Error during prediction")
                    st.exception(e)
        
    if 'df_churn_prediction' in st.session_state and st.session_state.df_churn_prediction is not None:
        with st.expander("Customer Segmentation with Clusters"):
            st.dataframe(st.session_state.df_churn_prediction.head(100))
        visualize_customer_churn(st.session_state.df_churn_prediction)

def render_segmentation_page():
    """Render the Customer Segmentation page"""
    st.title("👥 Customer Segmentation")
    st.markdown("Explore your customers based on their behavior and preferences")
    st.divider()
    back_to_home_button()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload with caching
        uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=['csv', 'xlsx'], key="segment_file")
        df = process_uploaded_file(uploaded_file)
    
    with col2:
        with st.expander("What data do I need?"):
            st.info("""
            #### Input Dataset Requirements
            Your dataset should include at least these **5 columns**:

            - `Customer_ID`  
            - `Customer_Name`  
            - `PurchaseDate`  
            - `Price`  
            - `Quantity`  

            #### Output After Processing
            The system will calculate:

            - `Last Purchase Date`  
            - `TotalSpent` (Monetary)  
            - `Frequency`  
            - `Recency`  
            - `Customer Labels`
            """)
    
    if df is not None:
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head(100))
        predict_button = st.button("🧠 Predict Clusters", use_container_width=True)
        if predict_button:
            with st.spinner("Generating clusters..."):
                try:
                    # Reset file position for reading
                    uploaded_file.seek(0)
                    files = {'file': uploaded_file}
                    response = requests.post(SEGMENT_API, files=files)
                    if response:
                        response_data = response.json()
                    df_cluster = pd.DataFrame(response_data["results"])
                    st.session_state.df_cluster = df_cluster
                    
                    with st.expander("Customer Segmentation Results", expanded=True):
                        st.dataframe(df_cluster.head(100))
                    
                    st.success("Customer Segmentation completed successfully!")
                    
                except Exception as e:
                    st.error("Something went wrong during segmentation.")
                    st.exception(e)
        
        if ("df_cluster" in st.session_state and predict_button):
            tabs = st.tabs(["Visualizations", "Business Insights"])
            
            with tabs[0]:
                visualize_customer_segments(st.session_state.df_cluster)
            
            with tabs[1]:
                st.info("""
                **Key Business Insights 💡**  
                - **VIP Segment 💎 (Cluster 0)**:  
                    - Represents the most valuable customers.  
                    - Should be prioritized for retention.  
                - **Lapsed Customers 😴 (Cluster 1)**:  
                    - Includes customers who have lost recent interaction with the shop.  
                    - Indicates a need for targeted retention strategies.  
                - **Regular Customers 🌱 (Cluster 2)**:  
                    - Show potential for upgrading to VIP status with proper nurturing.  

                **Recommended Actions 💰**  
                - **For VIPs**:  
                    - Implement a loyalty program.  
                    - Provide exclusive benefits.  
                - **For Lapsed Customers**:  
                    - Create win-back campaigns with attractive offers.  
                - **For Regular Customers**:  
                    - Develop up-selling and cross-selling strategies.  
                """)

def render_feedback_analysis_page():
    """Render the Customer Feedback Analysis page"""
    st.title("😊😡 Customer Feedback Analysis")
    st.markdown("**Explore your customers based on their feedbacks!**")
    st.divider()
    back_to_home_button()
    
    # Analysis options
    st.markdown("## Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Test Single Review", use_container_width=True):
            st.session_state.show_text_input = True
            st.session_state.show_batch_analysis = False
    
    with col2:
        if st.button("📂 Analyze Multiple Reviews", use_container_width=True):
            st.session_state.show_text_input = False
            st.session_state.show_batch_analysis = True
    
    # Single Review Analysis
    if st.session_state.show_text_input:
        st.subheader("Single Review Analysis")
        text = st.text_area('Enter your review:')
        if st.button("Analyze Text"):
            if text:
                with st.status("Analyzing..."):
                    response = requests.post(
                        SENTIMENT_API,
                        json={"text": text}
                    )
                    response.raise_for_status()
                    time.sleep(1.5)
                    if response:
                        result = response.json()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'**Sentiment:** {result["sentiment"]}')
                    with col2:
                        st.markdown(f'**Estimated rating:** {result["rating"]}')
            else:
                st.warning('Please enter text to analyze.')
    
    # Batch Review Analysis
    if st.session_state.show_batch_analysis:
        st.subheader("Batch Review Analysis")
        
        # File upload with caching
        uploaded_file = st.file_uploader("Choose a file to analyze", type=['csv', 'xlsx'], key="sentiment_file")
        if uploaded_file:
            df = process_uploaded_file(uploaded_file)
            
            if df is not None:
                st.success("Data loaded successfully!")
                with st.expander("Preview Data", expanded=True):
                    st.dataframe(df.head(100))
                    
                analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
                if analyze_button:
                    with st.spinner("Analyzing Reviews..."):
                        uploaded_file.seek(0)
                        files = {"file": uploaded_file}
                        response = requests.post(
                            SENTIMENT_BATCH_API,
                            files=files
                        )
                        if response:
                            batch_results = response.json()

                        # Convert API results back to DataFrame
                        df_results = pd.DataFrame(batch_results["results"])
                        st.session_state.df_sentiment_report = df_results
                        
                        with st.expander("View Analyzed Data", expanded=True):
                            st.dataframe(df_results.head(100))
                        st.success("Analysis completed!")
                    
        if 'df_sentiment_report' in st.session_state and st.session_state.df_sentiment_report is not None:
            st.subheader("Visualizations")
            viz_tabs = st.tabs(["Word Cloud", "Sentiment Distribution"])
            with viz_tabs[0]:
                st.markdown("### Word Cloud")
                word_cloud(st.session_state.df_sentiment_report)
            with viz_tabs[1]:
                st.markdown("### Sentiment Distribution")
                sentiment_distribution(st.session_state.df_sentiment_report)
