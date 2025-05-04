import streamlit as st
from FRONTEND.page import render_home_page, render_sidebar
from FRONTEND.components import render_chatbot_page, render_data_analyst_page, render_forecasting_page, render_segmentation_page, render_feedback_analysis_page, render_churn_page
from FRONTEND.state import init_session_state
from FRONTEND.utils import configure_page
configure_page(
    page_title="AI Decision-Making System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_session_state()

render_sidebar()

if st.session_state["page"] == "🏠 Home":
    render_home_page()
elif st.session_state["page"] == "Chatbot":
    render_chatbot_page()
elif st.session_state["page"] == "AI Data Analyst":
    render_data_analyst_page()
elif st.session_state["page"] == "Forecasting Profit":
    render_forecasting_page()
elif st.session_state["page"] == "Customer Segmentation":
    render_segmentation_page()
elif st.session_state["page"] == "Customer Feedback Analysis":
    render_feedback_analysis_page()
elif st.session_state["page"] == "Predict Churn Probability":
    render_churn_page()
else:
    render_home_page()
