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
st.set_page_config(
    page_title="AI Decision-Making System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

# Sidebar configuration
st.sidebar.title("**🤖 CORTEX**")
st.sidebar.title("Our Features ⚙️")

# Feature Navigation in Sidebar with expanders
with st.sidebar.expander("💬 Multi Agent AI Chatbot"):
    st.markdown("""
    - Ask anything about your data  
    - Use AI to generate answers or summaries
    """)
    if st.button("Go to Chatbot"):
        st.session_state["page"] = "Chatbot"
        st.rerun()

with st.sidebar.expander("📊 🤖 AI Data Analyst"):
    st.markdown("""
    - Upload financial or customer data  
    - Automatically generate insights, KPIs, and charts  
    """)
    if st.button("Go to AI Data Analyst"):
        st.session_state["page"] = "AI Data Analyst"
        st.rerun()

with st.sidebar.expander("📈 Forecasting Profit"):
    st.markdown("""
    - Predict future profit trends  
    - Plan marketing & spending wisely
    """)
    if st.button("Go to Profit Forecasting"):
        st.session_state["page"] = "Forecasting Profit"
        st.rerun()

with st.sidebar.expander("👥 Customer Segmentation"):
    st.markdown("""
    - Cluster customers based on behavior  
    - Find patterns and target precisely
    """)
    if st.button("Go to Segmentation"):
        st.session_state["page"] = "Customer Segmentation"
        st.rerun()

with st.sidebar.expander("😊😡 Feedback Analysis"):
    st.markdown("""
    - Perform sentiment analysis  
    - Know how customers feel
    """)
    if st.button("Go to Feedback Analysis"):
        st.session_state["page"] = "Customer Feedback Analysis"
        st.rerun()

with st.sidebar.expander("🔁 Predict Churn Probability"):
    st.markdown("""
    - Identify who might leave  
    - Take action before churn happens
    """)
    if st.button("Go to Churn Prediction"):
        st.session_state["page"] = "Predict Churn Probability"
        st.rerun()

st.sidebar.success("Use the buttons above to explore each feature 👉")

# Main content area
if st.session_state["page"] == "🏠 Home":
    st.title("**🤖 CORTEX - AI Decision-Making System**")
    st.markdown("### 📊 **Unlock Insights, Automate Reports**")
    st.markdown("Empower your business with cutting-edge AI Agentic and Machine Learning.")
    
    st.markdown("---")
    tab1, tab2 = st.tabs(["👤 ABOUT ME", "🧠 ABOUT SYSTEM"])
    with tab1:
            st.markdown("""
            <div style="text-align: center;">
                <h2 style="margin-bottom: 5px;"><strong>Nguyen Tran Trung</strong></h2>
                <p style="margin-top: 0;">National Economics University</p>
                <h4 style="color: grey; font-weight: normal;">Project Owner</h4>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🎯 Main Roles")
            st.markdown("""
            - 🧠 **AI Engineer**: Design & build Multi-AI Agent Systems  
            - 📊 **Data Scientist**: Train and deploy ML/DL models  
            """)

            st.markdown("### 🛠️ Supporting Roles")
            st.markdown("""
            - ⚙️ **Backend Engineer**: Connect frontend-backend via FastAPI  
            - 📈 **Data Analyst**: Data preprocessing & schema design  
            - 🧩 **Business Analyst**: Map business goals to technical strategies, design workflows, and write documentation  
            """)

            st.markdown("""
            <hr style="border-top: 1px solid #bbb;" />
            <p style="text-align:center;">
                🔗 <a href="https://github.com/Teungtran/AI-DECISION-MAKING-SYSTEM" target="_blank"><strong>More About This Project</strong></a>
            </p>
            """, unsafe_allow_html=True)
    with tab2:
            st.markdown("""
            <div style="text-align: center;">
                <h2><strong>AI Decision-Making System</strong></h2>
                <p style="margin-top: 0;"><em>Smart Automation for Business Operations & Strategy</em></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🔧 Key Modules")

            st.markdown("#### 📌 Business Operation Management")
            st.markdown("""
            - Answer questions about Amazon policies & processes  
            - Query structured business data (e.g., customers, products)  
            - Update insights via web search  
            - Personalize user interaction  
            - Recommend strategies & brainstorm ideas  
            """)

            st.markdown("#### 📊 Performance & Reporting")
            st.markdown("""
            - Generate on-demand business reports (Excel or DB)  
            - Visualize data interactively  
            - Real-time revenue forecasting  
            - Monitor sales, trends, and buyer behavior  
            """)

            st.markdown("#### 👥 Customer Management")
            st.markdown("""
            - Segment customers for engagement strategy  
            - Analyze sentiment from feedback  
            - Predict churn risk and suggest actions  
            """)
#____________________________FEATURES_______________________________________
if st.session_state["page"] == "AI Data Analyst":
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))

    st.session_state.current_feature = "AI Data Analyst"
    st.title(" AI Data Analyst📊")
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
    st.title("🛒 Cortex AI Chatbot 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))

    # Collapsible User Settings & Chat Tools
    with st.expander("User Settings & Chat Tools", expanded=True):
        session_id_input = st.text_input("User ID (optional):", value=st.session_state.session_id)
        if session_id_input:
            st.session_state.session_id = session_id_input

        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    with st.expander("💡 Try asking:", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📦 What is Amazon policy on selling platform for sellers"):
                prompt = "What is Amazon policy on selling platform for sellers"
        with col2:
            if st.button("👑 What is the number of VIP Customers labeled, and their average money spent"):
                prompt = "What is the number of VIP Customers labeled, and their average money spent"

        col3, col4 = st.columns(2)
        with col3:
            if st.button("📣 How can I set up a marketing campaign aimed at teen customers"):
                prompt = "How can I set up a marketing campaign aimed at teen customers"
        with col4:
            if st.button("📊 What is today's latest business trends"):
                prompt = "What is today's latest business trends"

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
                        response.raise_for_status()
                        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                            if chunk:
                                accumulated_response += chunk
                                response_container.markdown(accumulated_response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": accumulated_response
                    })

                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")


                    
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
    st.header("⚠️💸 Predict Churn Probability")
    st.sidebar.button("← Back to Home", on_click=lambda: st.session_state.update({"page": "🏠 Home"}))

    st.session_state.current_feature = "Predict Churn Probability"

    st.title("🔍📉🤔 Customer slipping away? Not on our watch! 👀🚀")
    uploaded_file = st.file_uploader("Upload your **CSV** or **XLSX** file", type=['csv', 'xlsx'])
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

    if uploaded_file is not None:
        try:
            df = import_data(uploaded_file)
            with st.expander("📄 Data Preview"):
                st.dataframe(df.head(100))

            if st.button("🚀 Predict Churn Probability"):
                with st.spinner("Running churn prediction..."):
                    try:
                        uploaded_file.seek(0)
                        files = {'file': uploaded_file}
                        response = requests.post(CHURN_API, files=files)

                        if response.status_code == 200:
                            response_data = response.json()
                            df_churn_prediction = pd.DataFrame(response_data["results"])
                            with st.expander("📊 Churn Probability Results"):
                                st.dataframe(df_churn_prediction.head(100))
                            st.success("✅ Churn Prediction completed!")
                            st.session_state.df_churn_prediction = df_churn_prediction
                        else:
                            st.error(f"API error {response.status_code}")
                            st.write(f"Details: {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error("Network error")
                        st.exception(e)
                    except Exception as e:
                        st.error("Error during prediction")
                        st.exception(e)
        except Exception as e:
            st.error("Error loading file")
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
    st.title("Explore your customers based on their behavior and preferences")

    uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=['csv', 'xlsx'])

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

    if uploaded_file is not None:
        df = import_data(uploaded_file)
        with st.expander("Data Preview"):
            st.dataframe(df.head(100))

        if df is not None:
            if st.button("Predict Clusters"):
                with st.spinner("Generating..."):
                    try:
                        uploaded_file.seek(0)
                        files = {'file': uploaded_file}
                        response = requests.post(SEGMENT_API, files=files)

                        if response.status_code == 200:
                            response_data = response.json()
                            df_cluster = pd.DataFrame(response_data["results"])

                            with st.expander("Customer Segmentation Results"):
                                st.dataframe(df_cluster.head(100))

                            st.session_state.df_cluster = df_cluster
                            st.success("Customer Segmentation completed successfully!")
                        else:
                            st.error(f"API error: {response.status_code}")
                            st.write(f"Response: {response.text}")
                    except Exception as e:
                        st.error("Something went wrong during segmentation.")
                        st.exception(e)
                        
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

    st.markdown("## Analysis Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Test Single Review"):
            st.session_state.show_text_input = True
            st.session_state.show_batch_analysis = False

    with col2:
        if st.button("📂 Analyze Multiple Reviews"):
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
