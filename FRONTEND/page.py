import streamlit as st
from FRONTEND.utils import feature_button
from FRONTEND.state import update_page

def render_sidebar():
    """Render the sidebar with navigation options"""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🤖 CORTEX</div>', unsafe_allow_html=True)
        st.markdown("### Our Features ⚙️")
        
        # Create expandable sections for each feature with styled buttons
        with st.expander("💬 Multi Agent AI Chatbot", expanded=False):
            st.markdown("""
            - Ask anything about your data  
            - Use AI to generate answers or summaries
            """)
            st.button("Go to Chatbot", key="sb_chatbot", 
                      on_click=update_page, args=("Chatbot",))
        
        with st.expander("📊 AI Data Analyst", expanded=False):
            st.markdown("""
            - Upload financial or customer data  
            - Automatically generate insights, KPIs, and charts  
            """)
            st.button("Go to AI Data Analyst", key="sb_analyst", 
                      on_click=update_page, args=("AI Data Analyst",))
        
        with st.expander("📈 Forecasting Profit", expanded=False):
            st.markdown("""
            - Predict future profit trends  
            - Plan marketing & spending wisely
            """)
            st.button("Go to Profit Forecasting", key="sb_forecast", 
                      on_click=update_page, args=("Forecasting Profit",))
        
        with st.expander("👥 Customer Segmentation", expanded=False):
            st.markdown("""
            - Cluster customers based on behavior  
            - Find patterns and target precisely
            """)
            st.button("Go to Segmentation", key="sb_segment", 
                      on_click=update_page, args=("Customer Segmentation",))
        
        with st.expander("😊😡 Feedback Analysis", expanded=False):
            st.markdown("""
            - Perform sentiment analysis  
            - Know how customers feel
            """)
            st.button("Go to Feedback Analysis", key="sb_feedback", 
                      on_click=update_page, args=("Customer Feedback Analysis",))
        
        with st.expander("🔁 Predict Churn Probability", expanded=False):
            st.markdown("""
            - Identify who might leave  
            - Take action before churn happens
            """)
            st.button("Go to Churn Prediction", key="sb_churn", 
                      on_click=update_page, args=("Predict Churn Probability",))
        
        st.markdown("---")
        st.success("👆 Use the buttons above to explore each feature!")

def render_home_page():
    """Render the home page"""
    st.markdown('<h1 class="main-title">🤖 CORTEX - AI Decision-Making System</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 **Unlock Insights, Automate Reports**")
    st.markdown("Empower your business with cutting-edge AI Agentic and Machine Learning.")
    
    st.markdown("---")
    
    # Create tabs for About sections
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
    
    # Feature cards section
    st.markdown("---")
    st.markdown("## 🚀 Explore Our Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_button(
            icon="💬",
            label="Multi Agent AI Chatbot",
            description="Chat with AI about your business data and get insights",
            key="home_chatbot",
            target_page="Chatbot"
        )
        
        feature_button(
            icon="📈",
            label="Forecasting Profit",
            description="Predict future profit trends to plan marketing & spending",
            key="home_forecast",
            target_page="Forecasting Profit"
        )
        
        feature_button(
            icon="😊😡",
            label="Feedback Analysis",
            description="Analyze customer sentiments from reviews and feedback",
            key="home_feedback",
            target_page="Customer Feedback Analysis"
        )
    
    with col2:
        feature_button(
            icon="📊",
            label="AI Data Analyst",
            description="Get automated insights, KPIs, and charts from your data",
            key="home_analyst",
            target_page="AI Data Analyst"
        )
        
        feature_button(
            icon="👥",
            label="Customer Segmentation",
            description="Cluster customers based on behavior for targeted marketing",
            key="home_segment",
            target_page="Customer Segmentation"
        )
        
        feature_button(
            icon="🔁",
            label="Predict Churn Probability",
            description="Identify customers at risk of leaving and take action",
            key="home_churn",
            target_page="Predict Churn Probability"
        )