import streamlit as st
import uuid
import requests
import os
from ML_Model.Customer_segmentation import visualize_customer_segments
from ML_Model.sentiment_function import word_cloud,sentiment_distribution
from ML_Model.churn_probability import visualize_customer_churn
from ML_Model.Profit_Forecasting import visualize_forecast,visualize_report
from AI_data_analyst.chat_csv import *
from AI_data_analyst.data_processing import *
BASE_URL = "http://localhost:8888"
FORECAST_API = f"{BASE_URL}/forecast/"
CHATBOT_API = f"{BASE_URL}/chatbot/"
CHURN_API = f"{BASE_URL}/churn/"
SEGMENT_API = f"{BASE_URL}/segment/"
SENTIMENT_BATCH_API = f"{BASE_URL}/sentiment/batch/"
SENTIMENT_API = f"{BASE_URL}/sentiment/"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = None
# Page configuration
st.set_page_config(
    page_title="AI Decision-Making System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Our Features ⚙️")
st.sidebar.markdown("Use **Home** to explore features.")

st.sidebar.markdown("---")

st.sidebar.markdown("#### Project Owner")
st.sidebar.markdown("""
**Nguyen Tran Trung - National Economics University**

**Main Roles:**
- **AI Engineer**: Design & build Multi-AI Agent Systems  
- **Data Scientist**: Train and deploy ML/DL models  

**Other Roles:**
- Backend Engineer: Streamlined frontend-backend via FastAPI  
- Data Analyst: Data preprocessing & schema design  
- Business Analyst: Translate business goals into technical strategies, identify data-driven opportunities, design data flow, write doc and API requirement
-----------------------------------------------------------------------------------------------------------------
[**For more details, please click here**](https://github.com/Teungtran/AI-DECISION-MAKING-SYSTEM)

""")

# Set session state default
if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

# Home Page
if st.session_state["page"] == "🏠 Home":
    st.title("**🤖 CORTEX - AI Decision-Making System**")
    st.markdown("### 📊 **Unlock Insights, Automate Reports**")
    st.markdown("Empower your business with cutting-edge AI Agentic and Machine Learning.")

    with st.expander("💬 Multi Agent AI Chatbot"):
        st.markdown("""
        - Ask anything about your data  
        - Use AI to generate answers or summaries
        """)
        if st.button("Go to Chatbot"):
            st.session_state["page"] = "Chatbot"
            st.rerun()

    with st.expander("📊Analyse Reports"):
        st.markdown("""
        - Upload financial or customer data  
        - Automatically generate insights, KPIs, and charts  
        """)
        if st.button("Go to Report Analysis"):
            st.session_state["page"] = "Analyse Reports"
            

    with st.expander("📈 Forecasting Profit"):
        st.markdown("""
        - Predict future profit trends  
        - Plan marketing & spending wisely
        """)
        if st.button("Go to Profit Forecasting"):
            st.session_state["page"] = "Forecasting Profit"
            st.rerun()


    with st.expander("👥 Customer Segmentation"):
        st.markdown("""
        - Cluster customers based on behavior  
        - Find patterns and target precisely
        """)
        if st.button("Go to Segmentation"):
            st.session_state["page"] = "Customer Segmentation"
            st.rerun()

    with st.expander("😊😡 Feedback Analysis"):
        st.markdown("""
        - Perform sentiment analysis  
        - Know how customers feel
        """)
        if st.button("Go to Feedback Analysis"):
            st.session_state["page"] = "Customer Feedback Analysis"
            st.rerun()

    with st.expander("🔁 Predict Churn Probability"):
        st.markdown("""
        - Identify who might leave  
        - Take action before churn happens
        """)
        if st.button("Go to Churn Prediction"):
            st.session_state["page"] = "Predict Churn Probability"
            st.rerun()

    st.success("Use the buttons above to explore each feature 👉")

# Analyse Reports
if st.session_state["page"] == "Analyse Reports":
    st.header("🧾 Analyse Reports")
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))
    st.write("Your chatbot interface here.")

    st.session_state.current_feature = "Analyse Reports"
    st.title("Analyse Reports📊")
    st.markdown("**Explore your data in an interactive way!**")
    st.divider()
    df = None
    result = None
    uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        df = import_data(uploaded_file)
        with st.expander("Data Preview:"):  
            st.dataframe(df.head(100))
        if st.button("Convert Categorical data to Numeric"):
            df_converted = get_dummies(df)
            with st.expander("Data after getting dummies:"):
                st.dataframe(df_converted.head(100))
        if st.button("Analyze Dataset"):
            with st.spinner("Analyzing dataset..."):
                with st.expander("Dataset info"):
                    dataset_info = analyze_dataset(df)
                    st.text_area("Dataset Analysis", dataset_info, height=400)
        setup_chat_history()
        if df is not None:
            handle_chat_interface(df, GOOGLE_API_KEY, model)
        if st.sidebar.button("Clear Chat History"):
            clear_chat_history()
        st.sidebar.subheader("Select Columns and Plot Type")
        visualize_data(df)

elif st.session_state["page"] == "Chatbot":
    st.header("💬 Chatbot")
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.sidebar.markdown("<h2 style='color:white;'>🛒 Cortex AI Assistant</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='color:white;'>Categories</h3>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        - 📝 **Policy**  
        Get store policies and return/exchange info.  
        - 👥 **Customers**  
        Manage customer profiles & inquiries.  
        - 📦 **Orders**  
        Track orders, refunds, and shipments.  
        - 🔍 **Search**  
        Find answers to common questions.  
        """, 
        unsafe_allow_html=True
    )


    st.title("🛒 Cortex AI Assistant 🤖")

    session_id_input = st.sidebar.text_input("Enter user id (optional)")
    if session_id_input:
        st.session_state.session_id = session_id_input

    if st.sidebar.button("Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type your message here...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"user_input": prompt, "user_id": st.session_state.session_id}
                    accumulated_response = ""
                    response_container = st.empty()
                    
                    with requests.post(CHATBOT_API, json=payload, stream=True, timeout=30) as response:
                        response.raise_for_status()  # Raise exception for non-200 status
                        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                            if chunk:
                                accumulated_response += chunk
                                response_container.markdown(accumulated_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": accumulated_response})
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {str(e)}")
                except Exception as e:
                    st.error(f"General error: {str(e)}")
                    
elif st.session_state["page"] == "Forecasting Profit":
    st.header("📈📊 Forecasting Profit")
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))
    st.title("**🔍📈💰 Where's the money going? Let's find out! 💸💰**")
    st.session_state.current_feature = "Forecasting Profit"

    uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            df = import_data(uploaded_file)

            with st.expander("Data Preview"):
                st.dataframe(df.head(100))

            future_periods = st.number_input(
                "Number of periods to forecast (monthly)",
                value=365
            )

            # Store forecast output in session state to persist across reruns
            if st.button("Look into the Future!👀"):
                with st.spinner("Processing data..."):
                    try:
                        uploaded_file.seek(0)
                        files = {'file': uploaded_file}
                        payload = {'future_periods': future_periods}
                        response = requests.post(FORECAST_API, data=payload, files=files)

                        response.raise_for_status()
                        forecast_data = response.json()
                        st.session_state['forecast_data'] = forecast_data
                        st.session_state['original_df'] = df 
                        st.success("Forecasting completed successfully!")

                    except requests.exceptions.HTTPError as http_err:
                        st.error(f"HTTP error occurred: {http_err}")
                        st.json(response.json())
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
                        st.exception(e)

            # Show visualization if forecast_data exists
            if 'forecast_data' in st.session_state:
                forecast_data = st.session_state['forecast_data']
                forecast_df = pd.DataFrame(forecast_data['results'])
                
                # Convert to datetime index if results contain date strings
                if 'index' in forecast_df.columns:
                    forecast_df.set_index('index', inplace=True)
                    forecast_df.index = pd.to_datetime(forecast_df.index)
                
                start_date = forecast_data['start_date']
                end_date = forecast_data['end_date']
                start_date = pd.to_datetime(start_date)
                
                with st.expander("Forecasting Preview"):
                    st.dataframe(forecast_df)
                    st.markdown(f"**Forecast Start Date:** {start_date}")
                    st.markdown(f"**Forecast End Date:** {end_date}")

                st.subheader("Visualization")
                viz_tabs = st.tabs(['Forecasting', 'Report'])

                with viz_tabs[0]:
                    try:
                        visualize_forecast(forecast_df, start_date)
                    except Exception as e:
                        st.error(f"Error in forecast visualization: {str(e)}")
                        st.exception(e)
                        
                        # Display debug information
                        st.subheader("Debug Information")
                        st.write("Forecast DataFrame Info:")
                        st.write(f"Shape: {forecast_df.shape}")
                        st.write(f"Columns: {forecast_df.columns.tolist()}")
                        st.write(f"Index type: {type(forecast_df.index)}")
                        st.write(f"Split date: {start_date}")
                
                with viz_tabs[1]:
                    try:
                        if 'original_df' in st.session_state:
                            visualize_report(st.session_state['original_df'])
                        else:
                            st.warning("Original data not available for reporting. Please re-upload your file.")
                    except Exception as e:
                        st.error(f"Error in report visualization: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)


elif st.session_state["page"] == "Predict Churn Probability":
    st.header("⚠️💸Predict Churn Probability")
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))

    st.session_state.current_feature = "Predict Churn Probability"

    st.title("**🔍📉🤔 Customer slipping away? Not on our watch! 👀🚀**")
    st.sidebar.subheader("Instruction")
    st.divider()
    st.sidebar.info("""
                    **Note:** The model requires a dataset with these columns: 
                    - customer_id
                    - Purchase Date
                    - Product Category
                    - Product Price
                    - Quantity
                    - Payment Method
                    - Customer Name
                    - Customer_Labels
                    - Price
                    """)
    st.sidebar.info("""
                    **The model will calculate and give out the output of:** 
                    - Favoured_Product_Categories
                    - Frequency
                    - TotalSpent
                    - Favoured_Payment_Methods
                    - Customer_Label
                    - Recency
                    - Avg_Spend_Per_Purchase
                    - Purchase_Consistency
                    - Churn_Probability
                    """)
    uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            # Import and process data for preview
            df = import_data(uploaded_file)
            
            with st.expander("Data Preview:"):  
                st.dataframe(df.head(100))
                
            if df is not None:
                if st.button("Predict Churn Probability"):
                    with st.spinner("Predicting Churn Probability..."):
                        try:
                            # Reset file pointer to beginning
                            uploaded_file.seek(0)
                            
                            # Prepare the file for API upload
                            files = {'file': uploaded_file}
                            
                            # Make API request
                            response = requests.post(CHURN_API, files=files)
                            
                            
                            if response.status_code == 200:
                                # Parse response
                                response_data = response.json()
                                
                                # Since the backend now only returns churn_predictions
                                df_churn_prediction = pd.DataFrame(response_data["results"])
                                
                                with st.expander("Churn Probability Preview:"):
                                    st.dataframe(df_churn_prediction.head(100))
                                    
                                st.success("Churn Prediction completed successfully!")
                                # Store prediction in session state
                                st.session_state.df_churn_prediction = df_churn_prediction
                            else:
                                st.error(f"API request failed with status code {response.status_code}")
                                st.write(f"Response content: {response.text}")
                        
                        except requests.exceptions.RequestException as e:
                            st.error(f"Network error: {str(e)}")
                            st.exception(e)
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.exception(e)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)

    # Moved the visualization button section outside the file upload logic
    if "df_churn_prediction" in st.session_state:
        if st.button("Visualize Result"):
            with st.expander("Customer Segmentation with Clusters"):
                st.dataframe(st.session_state.df_churn_prediction.head(100))
            visualize_customer_churn(st.session_state.df_churn_prediction)
            
elif st.session_state["page"] == "Customer Segmentation":
    st.header("🧠 Customer Segmentation")
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))

    st.session_state.current_feature = "Customer Segmentation"
    st.title("**Explore your customers based on their behavior and preferences!**")
    
    st.sidebar.subheader("Instruction")
    st.divider()
    st.sidebar.info("""
                    **Note:** The model requires a dataset with at least 5 columns: 
                    - Customer_ID
                    - Customer_Name
                    - PurchaseDate
                    - Price
                    - Quantity
                    """)
    st.sidebar.info("""
                    **The model will calculate and give out the output of:** 
                    - Last Purchase Date
                    - TotalSpent (Monetary)
                    - Frequency 
                    - Recency 
                    - Customers Labels
                    """)
    uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        df = import_data(uploaded_file)
        with st.expander("Data Preview:"):  
            st.dataframe(df.head(100))
        if df is not None:
            if st.button("Predict Clusters"):
                with st.spinner("Generating..."):
                    uploaded_file.seek(0)      
                    files = {'file': uploaded_file}
                    response = requests.post(SEGMENT_API, files=files)
                    if response.status_code == 200:
                        response_data = response.json()
                        df_cluster = pd.DataFrame(response_data["results"])
                    with st.expander("Customer Segmentation with Clusters"):
                        st.dataframe(df_cluster.head(100))    
                    st.session_state.df_cluster = df_cluster                    
                    st.success("Customer Segmentation completed successfully!")

            if st.button("Visualize Result"):
                with st.expander("Customer Segmentation with Clusters"):
                    st.dataframe(st.session_state.df_cluster.head(100))
                visualize_customer_segments(st.session_state.df_cluster)  
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
elif st.session_state["page"] == "Customer Feedback Analysis":
    st.header("😊😡 Customer Feedback Analysis")
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))
    st.session_state.current_feature = "Customer Feedback Analysis"
    st.title("**Explore your customers based on their feedbacks!**")
    st.divider()

    # Initialize session state if not already initialized
    if "show_text_input" not in st.session_state:
        st.session_state.show_text_input = False
    if "show_batch_analysis" not in st.session_state:
        st.session_state.show_batch_analysis = False

    # Sidebar for selecting analysis type
    with st.sidebar:
        st.header("Analysis Options")
        if st.button("Test Single Review"):
            st.session_state.show_text_input = True
            st.session_state.show_batch_analysis = False
        if st.button("Analyze Multiple Reviews"):
            st.session_state.show_text_input = False
            st.session_state.show_batch_analysis = True

    # Single Review Analysis
    if st.session_state.show_text_input:
        text = st.text_area('Enter your review:')
        if st.button("Analyze Text"):
            if text:
                with st.status("Analyzing..."):
                    try:
                        response = requests.post(
                            SENTIMENT_API,
                            json={"text": text}
                        )
                        response.raise_for_status()
                        result = response.json()
                        time.sleep(1.5)
                        st.markdown(f'**Cortex believes that your review is:** {result["sentiment"]}')
                        st.markdown(f'**Estimated rating:** {result["rating"]}')
                    except requests.exceptions.RequestException as e:
                        st.error(f"API Error: {e}")
            else:
                st.warning('Please enter text to analyze.')

    # Batch Review Analysis
    if st.session_state.show_batch_analysis:
        uploaded_file = st.file_uploader("Choose a file to analyze", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            try:
                # Load data from uploaded file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format. Please upload CSV or Excel.")
                    df = None

                if df is not None:
                    st.success("Data loaded successfully!")
                    with st.expander("Preview Data", expanded=False):
                        st.dataframe(df.head())

                    if st.button("Analyze Sentiment"):
                        with st.spinner("Analyzing Reviews..."):
                            uploaded_file.seek(0)  # Reset file pointer
                            try:
                                files = {"file": uploaded_file}
                                response = requests.post(
                                    SENTIMENT_BATCH_API,
                                    files=files
                                )
                                response.raise_for_status()
                                batch_results = response.json()

                                # Convert API results back to DataFrame
                                df_results = pd.DataFrame(batch_results["results"])
                                st.session_state.df_sentiment_report = df_results
                                time.sleep(2)

                                with st.expander("View Analyzed Data", expanded=True):
                                    st.dataframe(st.session_state.df_sentiment_report.head(100))
                                st.success("Analysis completed!")
                            except requests.exceptions.RequestException as e:
                                st.error(f"API Error: {e}")
            except Exception as e:
                st.error(f'Error: {e}. Please check your data source and try again.')

        # Visualizations for Batch Analysis
        if 'df_sentiment_report' in st.session_state:
            if st.button("Generate Visualizations"):
                with st.expander("View Analyzed Data", expanded=True):
                    st.dataframe(st.session_state.df_sentiment_report.head(100))
                st.subheader("Visualizations")
                viz_tabs = st.tabs(["Word Cloud", "Sentiment Distribution"])
                with viz_tabs[0]:
                    st.markdown("### Word Cloud")
                    word_cloud(st.session_state.df_sentiment_report)
                with viz_tabs[1]:
                    st.markdown("### Sentiment Distribution")
                    sentiment_distribution(st.session_state.df_sentiment_report)
