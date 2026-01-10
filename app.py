"""
Sales Intelligence Copilot - Streamlit App
End-to-end analytics: Upload → Analyze → Download Report
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime
import io

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing import DataProcessor
from src.metrics import MetricsCalculator
from src.forecasting import train_and_forecast
from src.churn_model import train_and_predict_churn
from config.settings import TOP_N_AT_RISK, FORECAST_HORIZON_MONTHS

# Page configuration
st.set_page_config(
    page_title="Sales Intelligence Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None
    if 'churn_predictor' not in st.session_state:
        st.session_state.churn_predictor = None
    if 'at_risk_customers' not in st.session_state:
        st.session_state.at_risk_customers = None


def load_sample_data():
    """Load sample data for demo"""
    from config.settings import SAMPLE_DATA_DIR
    
    processor = DataProcessor()
    
    customers_file = SAMPLE_DATA_DIR / "customers.csv"
    subscriptions_file = SAMPLE_DATA_DIR / "subscriptions.csv"
    transactions_file = SAMPLE_DATA_DIR / "transactions.csv"
    
    if not customers_file.exists():
        st.error("Sample data not found. Please run `python data/generate_data.py` first.")
        return None
    
    with st.spinner("Loading sample data..."):
        processor.load_customers(str(customers_file))
        processor.load_subscriptions(str(subscriptions_file))
        processor.load_transactions(str(transactions_file))
        processor.merge_datasets()
    
    return processor


def upload_section():
    """File upload section"""
    st.markdown('<div class="main-header">📊 Sales Intelligence Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered sales forecasting and churn prediction</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Your Data")
        st.markdown("Upload your sales data files or use sample data to try the demo.")
        
        customers_file = st.file_uploader("Customers CSV", type=['csv', 'xlsx'], key='customers')
        subscriptions_file = st.file_uploader("Subscriptions CSV", type=['csv', 'xlsx'], key='subscriptions')
        transactions_file = st.file_uploader("Transactions CSV (Optional)", type=['csv', 'xlsx'], key='transactions')
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🚀 Load Data", type="primary", use_container_width=True):
                if customers_file and subscriptions_file:
                    try:
                        processor = DataProcessor()
                        
                        with st.spinner("Loading and processing data..."):
                            # Save uploaded files temporarily
                            customers_df = pd.read_csv(customers_file)
                            subscriptions_df = pd.read_csv(subscriptions_file)
                            
                            # Create temp files
                            temp_dir = Path("temp")
                            temp_dir.mkdir(exist_ok=True)
                            
                            customers_path = temp_dir / "customers.csv"
                            subscriptions_path = temp_dir / "subscriptions.csv"
                            
                            customers_df.to_csv(customers_path, index=False)
                            subscriptions_df.to_csv(subscriptions_path, index=False)
                            
                            processor.load_customers(str(customers_path))
                            processor.load_subscriptions(str(subscriptions_path))
                            
                            if transactions_file:
                                transactions_df = pd.read_csv(transactions_file)
                                transactions_path = temp_dir / "transactions.csv"
                                transactions_df.to_csv(transactions_path, index=False)
                                processor.load_transactions(str(transactions_path))
                            
                            processor.merge_datasets()
                        
                        st.session_state.processor = processor
                        st.session_state.data_loaded = True
                        st.success("✓ Data loaded successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                else:
                    st.warning("Please upload both Customers and Subscriptions files.")
        
        with col_b:
            if st.button("🎯 Use Sample Data", use_container_width=True):
                processor = load_sample_data()
                if processor:
                    st.session_state.processor = processor
                    st.session_state.data_loaded = True
                    st.success("✓ Sample data loaded!")
                    st.rerun()
    
    with col2:
        st.info("""
        **📋 Required Data Format:**
        
        **Customers CSV:**
        - customer_id
        - signup_date
        - plan (basic/pro/enterprise)
        - monthly_price
        - segment
        
        **Subscriptions CSV:**
        - subscription_id
        - customer_id
        - period_start
        - period_end
        - revenue
        - active
        - churn_flag
        """)


def data_preview_section():
    """Show data preview and quality report"""
    st.markdown("---")
    st.subheader("📊 Data Preview & Quality Report")
    
    processor = st.session_state.processor
    summary = processor.get_summary()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{summary['customers_count']:,}")
    with col2:
        st.metric("Subscription Records", f"{summary['subscriptions_count']:,}")
    with col3:
        st.metric("Active Subscriptions", f"{summary.get('active_subscriptions', 0):,}")
    with col4:
        st.metric("Total Revenue", f"${summary.get('total_revenue', 0):,.0f}")
    
    # Quality scores
    st.markdown("**Data Quality Scores:**")
    quality_cols = st.columns(len(summary['quality_scores']))
    for idx, (name, score) in enumerate(summary['quality_scores'].items()):
        with quality_cols[idx]:
            st.metric(name.title(), f"{score}%", delta=None)
    
    # Data preview tabs
    tab1, tab2, tab3 = st.tabs(["Customers", "Subscriptions", "Merged Data"])
    
    with tab1:
        st.dataframe(processor.customers.head(100), use_container_width=True)
    
    with tab2:
        st.dataframe(processor.subscriptions.head(100), use_container_width=True)
    
    with tab3:
        st.dataframe(processor.merged_data.head(100), use_container_width=True)
    
    # Run Analysis button
    st.markdown("---")
    if st.button("🤖 Run AI Analysis", type="primary", use_container_width=True):
        run_analysis()


def run_analysis():
    """Run forecasting and churn prediction"""
    processor = st.session_state.processor
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Calculate Metrics
        status_text.text("📊 Calculating business metrics...")
        progress_bar.progress(20)
        
        calculator = MetricsCalculator(processor.subscriptions, processor.customers)
        metrics = calculator.get_dashboard_metrics()
        st.session_state.metrics = metrics
        
        # Step 2: Train Forecasting Model
        status_text.text("🔮 Training forecasting model...")
        progress_bar.progress(40)
        
        forecaster, future_forecast = train_and_forecast(
            metrics['mrr_series'],
            forecast_months=FORECAST_HORIZON_MONTHS,
            save_model=False
        )
        st.session_state.forecaster = forecaster
        st.session_state.future_forecast = future_forecast
        
        # Step 3: Train Churn Model
        status_text.text("🎯 Training churn prediction model...")
        progress_bar.progress(60)
        
        churn_predictor, at_risk = train_and_predict_churn(
            processor.merged_data,
            model_type='logistic',
            save_model=False
        )
        st.session_state.churn_predictor = churn_predictor
        st.session_state.at_risk_customers = at_risk
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✓ Analysis complete!")
        
        st.session_state.analysis_complete = True
        st.balloons()
        st.rerun()
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def results_section():
    """Display analysis results"""
    st.markdown("---")
    st.markdown('<div class="main-header">📈 Analysis Results</div>', unsafe_allow_html=True)
    
    metrics = st.session_state.metrics
    
    # KPI Cards
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current MRR",
            f"${metrics['current_mrr']:,.0f}",
            delta=f"{metrics['mrr_growth_mom']:+.1f}% MoM"
        )
    
    with col2:
        st.metric("Annual ARR", f"${metrics['arr']:,.0f}")
    
    with col3:
        st.metric("Active Customers", f"{metrics['active_customers']:,}")
    
    with col4:
        st.metric("ARPU", f"${metrics['arpu']:,.2f}")
    
    with col5:
        st.metric(
            "Churn Rate",
            f"{metrics['churn_rate']:.1f}%",
            delta=f"{metrics['churn_rate'] - 5:.1f}%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Charts in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Revenue Forecast", "⚠️ Churn Analysis", "💰 Revenue Trends"])
    
    with tab1:
        show_forecast_chart()
    
    with tab2:
        show_churn_analysis()
    
    with tab3:
        show_revenue_trends()
    
    # Export section
    st.markdown("---")
    show_export_section()


def show_forecast_chart():
    """Display MRR forecast chart"""
    st.subheader("MRR Forecast - Next 3 Months")
    
    forecaster = st.session_state.forecaster
    forecast_summary = forecaster.get_forecast_summary()
    
    # Create Plotly chart
    fig = go.Figure()
    
    # Actual values
    actual_data = forecast_summary[forecast_summary['actual'].notna()]
    fig.add_trace(go.Scatter(
        x=actual_data['date'],
        y=actual_data['actual'],
        mode='lines',
        name='Actual MRR',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=forecast_summary['date'],
        y=forecast_summary['predicted'],
        mode='lines',
        name='Predicted MRR',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_summary['date'],
        y=forecast_summary['upper_bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_summary['date'],
        y=forecast_summary['lower_bound'],
        mode='lines',
        name='Lower Bound',
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(width=0),
        showlegend=True
    ))
    
    fig.update_layout(
        title="MRR Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="MRR ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    metrics = forecaster.metrics
    
    with col1:
        st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
    with col2:
        st.metric("MAE", f"${metrics.get('MAE', 0):,.2f}")
    with col3:
        st.metric("RMSE", f"${metrics.get('RMSE', 0):,.2f}")


def show_churn_analysis():
    """Display churn analysis"""
    st.subheader("High-Risk Customers")
    
    at_risk = st.session_state.at_risk_customers
    
    # Display at-risk customers table
    display_df = at_risk.head(TOP_N_AT_RISK).copy()
    display_df['churn_probability'] = (display_df['churn_probability'] * 100).round(1)
    display_df['current_revenue'] = display_df['current_revenue'].round(2)
    display_df['risk_score'] = display_df['risk_score'].round(2)
    
    st.dataframe(
        display_df[['customer_id', 'churn_probability', 'current_revenue', 'risk_score', 'tenure_months']],
        use_container_width=True,
        column_config={
            "customer_id": "Customer ID",
            "churn_probability": st.column_config.NumberColumn("Churn Risk (%)", format="%.1f%%"),
            "current_revenue": st.column_config.NumberColumn("Monthly Revenue", format="$%.2f"),
            "risk_score": st.column_config.NumberColumn("Risk Score", format="%.2f"),
            "tenure_months": st.column_config.NumberColumn("Tenure (Months)", format="%.1f")
        }
    )
    
    # Model performance
    st.markdown("**Model Performance:**")
    predictor = st.session_state.churn_predictor
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AUC", f"{predictor.metrics.get('auc', 0):.3f}")
    with col2:
        st.metric("Precision", f"{predictor.metrics.get('precision', 0):.3f}")
    with col3:
        st.metric("Recall", f"{predictor.metrics.get('recall', 0):.3f}")
    with col4:
        st.metric("F1 Score", f"{predictor.metrics.get('f1', 0):.3f}")
    
    # Recommendations
    st.markdown("**💡 Recommended Actions:**")
    st.markdown("""
    - 🎯 Reach out to top 5 at-risk customers with retention offers
    - 💰 Consider targeted discounts for high-revenue at-risk customers
    - 📞 Schedule customer success calls for customers with >70% churn probability
    - 📊 Analyze common patterns among at-risk customers to improve product
    """)


def show_revenue_trends():
    """Display revenue trend charts"""
    st.subheader("Revenue & Customer Trends")
    
    metrics = st.session_state.metrics
    
    # MRR over time
    fig_mrr = px.line(
        metrics['mrr_series'],
        x='date',
        y='MRR',
        title='MRR Trend'
    )
    fig_mrr.update_layout(height=400)
    st.plotly_chart(fig_mrr, use_container_width=True)
    
    # Churn rate over time
    fig_churn = px.line(
        metrics['churn_series'],
        x='date',
        y='churn_rate',
        title='Monthly Churn Rate (%)'
    )
    fig_churn.update_layout(height=400)
    st.plotly_chart(fig_churn, use_container_width=True)


def show_export_section():
    """Export and download options"""
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download at-risk customers CSV
        at_risk = st.session_state.at_risk_customers
        csv = at_risk.to_csv(index=False)
        st.download_button(
            label="📄 Download At-Risk Customers CSV",
            data=csv,
            file_name=f"at_risk_customers_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download forecast CSV
        forecast_summary = st.session_state.forecaster.get_forecast_summary()
        csv = forecast_summary.to_csv(index=False)
        st.download_button(
            label="📈 Download Forecast CSV",
            data=csv,
            file_name=f"mrr_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        if st.button("📊 Generate PPT Report", use_container_width=True):
            with st.spinner("Generating PowerPoint report..."):
                try:
                    from src.insights_generator import generate_insights
                    from src.report_builder import build_full_report
                    
                    # Add forecast_mrr to metrics for insights
                    metrics = st.session_state.metrics.copy()
                    metrics['forecast_mrr'] = st.session_state.future_forecast['predicted'].iloc[-1]
                    metrics['at_risk_count'] = len(st.session_state.at_risk_customers)
                    
                    # Generate insights
                    insights = generate_insights(
                        metrics,
                        st.session_state.at_risk_customers
                    )
                    
                    # Build report
                    filepath = build_full_report(
                        metrics,
                        st.session_state.future_forecast,
                        st.session_state.at_risk_customers,
                        insights,
                        st.session_state.forecaster
                    )
                    
                    st.success(f"✓ Report generated!")
                    
                    # Download button
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="📥 Download PowerPoint",
                            data=f,
                            file_name=f"sales_intelligence_report_{datetime.now().strftime('%Y%m%d')}.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                        )
                        
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")


def main():
    """Main app flow"""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.image("assets\project logo.png", use_container_width=True)
        st.markdown("---")
        
        st.markdown("### 📊 Navigation")
        if not st.session_state.data_loaded:
            st.info("👆 Upload data to begin")
        elif not st.session_state.analysis_complete:
            st.info("✓ Data loaded\n\n👇 Run analysis")
        else:
            st.success("✓ Analysis complete!")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Sales Intelligence Copilot** uses AI to:
        - Forecast revenue trends
        - Predict customer churn
        - Identify at-risk customers
        - Generate actionable insights
        """)
        
        if st.button("🔄 Reset App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    if not st.session_state.data_loaded:
        upload_section()
    elif not st.session_state.analysis_complete:
        data_preview_section()
    else:
        results_section()


if __name__ == "__main__":
    main()