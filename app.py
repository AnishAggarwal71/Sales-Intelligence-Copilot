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
from config.settings import TOP_N_AT_RISK, FORECAST_HORIZON_MONTHS, HIGH_RISK_THRESHOLD

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
    .metric-hint {
        font-size: 0.75rem;
        color: #888;
        margin-top: -0.8rem;
        margin-bottom: 1rem;
    }
    .step-indicator {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        margin: 0.5rem 0;
    }
    .step-done   { color: #2ca02c; font-weight: 600; }
    .step-active { color: #1f77b4; font-weight: 600; }
    .step-todo   { color: #aaa; }
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
    if 'future_forecast' not in st.session_state:
        st.session_state.future_forecast = None
    if 'churn_predictor' not in st.session_state:
        st.session_state.churn_predictor = None
    if 'at_risk_customers' not in st.session_state:
        st.session_state.at_risk_customers = None
    if 'generated_report_bytes' not in st.session_state:
        st.session_state.generated_report_bytes = None
    if 'generated_report_name' not in st.session_state:
        st.session_state.generated_report_name = None
    if 'validation_report' not in st.session_state:
        st.session_state.validation_report = {}


def _fmt(value, fmt=",.0f", fallback="—"):
    """Format a numeric value safely, returning fallback if NaN/Inf/None."""
    try:
        import math
        if value is None or math.isnan(float(value)) or math.isinf(float(value)):
            return fallback
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return fallback


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
        if transactions_file.exists():
            processor.load_transactions(str(transactions_file))
        processor.merge_datasets()

    return processor


def save_uploaded_file(uploaded_file, temp_dir: Path, base_name: str) -> Path:
    """Persist an uploaded CSV/XLSX file locally while preserving its extension."""
    suffix = Path(uploaded_file.name).suffix.lower() or '.csv'
    temp_path = temp_dir / f"{base_name}{suffix}"
    temp_path.write_bytes(uploaded_file.getvalue())
    return temp_path


def show_validation_report(report: dict):
    """
    Render a tiered data-validation panel.
    - Required missing  → red block, stops the user
    - Derivable missing → blue info card showing name, description, formula, example
    - Optional missing  → grey note (degraded accuracy)
    - All found         → green summary
    """
    from src.data_processing import COLUMN_REGISTRY

    file_labels = {'customers': 'Customers file', 'subscriptions': 'Subscriptions file'}

    for file_type, result in report.items():
        label = file_labels.get(file_type, file_type.title())
        st.markdown(f"**{label}**")

        # ── Found columns ──────────────────────────────────────────────────
        if result['found']:
            st.success(f"✓ Found {len(result['found'])} recognised columns: "
                       f"`{'`, `'.join(sorted(result['found']))}`")

        # ── Critically missing ─────────────────────────────────────────────
        if result['required_missing']:
            for col, desc in result['required_missing'].items():
                st.error(
                    f"**Missing required column: `{col}`**\n\n"
                    f"{desc}\n\n"
                    f"This column cannot be calculated automatically. "
                    f"Please add it to your file before uploading."
                )

        # ── Auto-derived ───────────────────────────────────────────────────
        if result['derivable_missing']:
            with st.expander(
                f"ℹ️ {len(result['derivable_missing'])} column(s) not found — "
                f"calculated automatically for you",
                expanded=True
            ):
                for col, meta in result['derivable_missing'].items():
                    st.markdown(f"""
<div style="border-left: 4px solid #1f77b4; padding: 0.6rem 1rem; margin-bottom: 0.8rem; background: #f0f6ff; border-radius: 0 4px 4px 0;">
<strong><code>{col}</code></strong><br>
<span style="color:#444;">{meta['description']}</span><br><br>
<strong>How it's calculated:</strong> {meta['calculation']}<br>
<strong>Example:</strong> <em>{meta['example']}</em>
</div>
""", unsafe_allow_html=True)

        # ── Optional missing ───────────────────────────────────────────────
        if result['optional_missing']:
            with st.expander(
                f"⚠️ {len(result['optional_missing'])} optional column(s) not found "
                f"— analysis will still run, accuracy may be reduced"
            ):
                for col, desc in result['optional_missing'].items():
                    st.markdown(f"- **`{col}`** — {desc}")

        st.markdown("")  # spacer between files


def upload_section():
    """File upload section"""
    st.markdown('<div class="main-header">📊 Sales Intelligence Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your subscription data → get revenue forecasts, churn predictions, and an AI-generated report — in under 2 minutes.</div>', unsafe_allow_html=True)

    # How it works strip
    st.markdown(
        "**How it works:** &nbsp; "
        "**1️⃣ Upload** your Customers + Subscriptions CSV &nbsp;→&nbsp; "
        "**2️⃣ Run Analysis** (AI trains in ~30 seconds) &nbsp;→&nbsp; "
        "**3️⃣ Download** your forecast, at-risk list, and PowerPoint report"
    )
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Your Data")
        st.markdown("Upload your sales data files or use sample data to try the demo.")
        
        customers_file = st.file_uploader("Customers file", type=['csv', 'xlsx', 'xls'], key='customers')
        subscriptions_file = st.file_uploader("Subscriptions file", type=['csv', 'xlsx', 'xls'], key='subscriptions')
        transactions_file = st.file_uploader("Transactions file (Optional)", type=['csv', 'xlsx', 'xls'], key='transactions')
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🚀 Load Data", type="primary", width='stretch'):
                if customers_file and subscriptions_file:
                    try:
                        processor = DataProcessor()

                        with st.spinner("Loading and validating data..."):
                            temp_dir = Path("temp")
                            temp_dir.mkdir(exist_ok=True)

                            customers_path = save_uploaded_file(customers_file, temp_dir, "customers")
                            subscriptions_path = save_uploaded_file(subscriptions_file, temp_dir, "subscriptions")

                            processor.load_customers(str(customers_path))
                            processor.load_subscriptions(str(subscriptions_path))

                            if transactions_file:
                                transactions_path = save_uploaded_file(transactions_file, temp_dir, "transactions")
                                processor.load_transactions(str(transactions_path))

                            # Build report BEFORE derivation so it reflects what the user uploaded
                            report = processor.build_validation_report()

                            # Block if any required columns are missing
                            blocking = [f for f, r in report.items() if not r['can_proceed']]
                            if blocking:
                                st.session_state.validation_report = report
                                st.rerun()

                            # Auto-fill derivable columns, then merge
                            processor.derive_missing_columns()
                            processor.merge_datasets()

                        st.session_state.processor = processor
                        st.session_state.validation_report = report
                        st.session_state.data_loaded = True
                        st.session_state.analysis_complete = False
                        st.session_state.generated_report_bytes = None
                        st.session_state.generated_report_name = None
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                else:
                    st.warning("Please upload both Customers and Subscriptions files.")
        
        with col_b:
            if st.button("🎯 Use Sample Data", width='stretch'):
                processor = load_sample_data()
                if processor:
                    st.session_state.processor = processor
                    st.session_state.data_loaded = True
                    st.session_state.analysis_complete = False
                    st.session_state.generated_report_bytes = None
                    st.session_state.generated_report_name = None
                    st.success("✓ Sample data loaded!")
                    st.rerun()
    
    with col2:
        st.markdown("**📋 Data Format Guide**")

        st.markdown("**Customers CSV**")
        st.markdown("""
| Column | Status |
|--------|--------|
| `customer_id` | ✅ Required |
| `signup_date` | ✅ Required |
| `monthly_price` | 🔵 Auto-calculated |
| `plan` | ⭐ Recommended |
| `segment` | ⭐ Recommended |
| `country` | ➕ Optional |
| `first_source` | ➕ Optional |
""")

        st.markdown("**Subscriptions CSV**")
        st.markdown("""
| Column | Status |
|--------|--------|
| `customer_id` | ✅ Required |
| `period_start` | ✅ Required |
| `period_end` | ✅ Required |
| `revenue` | ✅ Required |
| `churn_flag` | 🔵 Auto-calculated |
| `active` | 🔵 Auto-calculated |
| `is_renewal` | 🔵 Auto-calculated |
| `subscription_id` | 🔵 Auto-calculated |
| `num_logins` | ⭐ Recommended |
| `feature_x_usage` | ⭐ Recommended |
""")

        st.caption("✅ Required — app won't run without these  \n"
                   "🔵 Auto-calculated — derived from your data  \n"
                   "⭐ Recommended — improves churn prediction accuracy  \n"
                   "➕ Optional — used in reporting if provided")

        # If there was a blocking validation error, surface it here too
        report = st.session_state.get('validation_report', {})
        blocking = {f: r for f, r in report.items() if not r['can_proceed']}
        if blocking:
            st.markdown("---")
            st.markdown("**Issues found in your last upload:**")
            show_validation_report(blocking)


def data_preview_section():
    """Show data preview and quality report"""
    # ── Back button ────────────────────────────────────────────────────────
    if st.button("← Upload different files"):
        st.session_state.data_loaded = False
        st.session_state.processor = None
        st.session_state.validation_report = {}
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Data Preview & Quality Report")

    # ── Validation report ──────────────────────────────────────────────────
    report = st.session_state.get('validation_report', {})
    if report:
        # Surface blocking errors first
        blocking = {f: r for f, r in report.items() if not r['can_proceed']}
        if blocking:
            st.error("Your data is missing required columns. Please fix the issues below and re-upload.")
            show_validation_report(blocking)
            return

        # Check if anything interesting happened (derivations or optional gaps)
        has_derived  = any(r['derivable_missing'] for r in report.values())
        has_optional = any(r['optional_missing']  for r in report.values())

        if has_derived or has_optional:
            with st.expander("📋 Data Validation Report", expanded=has_derived):
                show_validation_report(report)
        else:
            st.success("✓ All expected columns found — no adjustments needed.")

    processor = st.session_state.processor
    summary = processor.get_summary()

    # ── Persistent load-success banner ────────────────────────────────────
    c_count = summary.get('customers_count', 0)
    s_count = summary.get('subscriptions_count', 0)
    if c_count == 0 or s_count == 0:
        st.warning("⚠️ Your files loaded but contain no records. Check that the files aren't empty and the column names match the expected format.")
    else:
        st.success(f"✓ Data loaded — **{c_count:,} customers** and **{s_count:,} subscription records** ready for analysis.")

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
    st.markdown("**Data Quality Scores** *(percentage of non-missing values in each file — higher is better)*")
    quality_cols = st.columns(len(summary['quality_scores']))
    for idx, (name, score) in enumerate(summary['quality_scores'].items()):
        with quality_cols[idx]:
            st.metric(name.title(), f"{score}%", delta=None)
    
    # Data preview tabs
    tab1, tab2, tab3 = st.tabs(["Customers", "Subscriptions", "Merged Data"])
    
    with tab1:
        st.dataframe(processor.customers.head(100), width='stretch')
    
    with tab2:
        st.dataframe(processor.subscriptions.head(100), width='stretch')
    
    with tab3:
        st.dataframe(processor.merged_data.head(100), width='stretch')
    
    # Run Analysis button
    st.markdown("---")
    if st.button("🤖 Run AI Analysis", type="primary", width='stretch'):
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
    # ── Back button ────────────────────────────────────────────────────────
    if st.button("← Re-run or change data"):
        st.session_state.analysis_complete = False
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="main-header">📈 Analysis Results</div>', unsafe_allow_html=True)

    metrics = st.session_state.metrics

    # KPI Cards
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Current MRR",
            f"${_fmt(metrics['current_mrr'])}",
            delta=f"{_fmt(metrics['mrr_growth_mom'], '+.1f')}% MoM"
        )
    with col2:
        st.metric("Annual ARR", f"${_fmt(metrics['arr'])}")
    with col3:
        st.metric("Active Customers", f"{_fmt(metrics['active_customers'], ',d')}")
    with col4:
        st.metric("Avg Revenue / Customer", f"${_fmt(metrics['arpu'], ',.2f')}")
    with col5:
        churn = metrics['churn_rate']
        churn_safe = churn if (churn == churn and churn != float('inf')) else 0.0
        st.metric(
            "Churn Rate",
            f"{_fmt(churn_safe, '.1f')}%",
            delta=f"{_fmt(churn_safe - 5, '+.1f')}% vs 5% benchmark",
            delta_color="inverse"
        )

    # Plain-English hints row
    st.markdown(
        '<div class="metric-hint">'
        'MRR = Monthly Recurring Revenue &nbsp;|&nbsp; '
        'ARR = MRR × 12 &nbsp;|&nbsp; '
        'Avg Revenue / Customer = total revenue ÷ unique billed customers &nbsp;|&nbsp; '
        'Churn Rate = % of customers who cancelled this month'
        '</div>',
        unsafe_allow_html=True
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
        name='Forecast range (80% confidence)',
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
    
    st.plotly_chart(fig, width='stretch')
    
    # Show metrics
    st.markdown("**Forecast Accuracy** *(how well the model predicted past months it hadn't seen)*")
    col1, col2, col3 = st.columns(3)
    metrics = forecaster.metrics

    with col1:
        st.metric("MAPE", f"{_fmt(metrics.get('MAPE'), '.2f')}%")
        st.caption("Mean Absolute % Error — lower is better. Under 15% is good.")
    with col2:
        st.metric("MAE", f"${_fmt(metrics.get('MAE'))}")
        st.caption("Average dollar error per month in the forecast.")
    with col3:
        st.metric("RMSE", f"${_fmt(metrics.get('RMSE'))}")
        st.caption("Like MAE but penalises large misses more heavily.")


def show_churn_analysis():
    """Display churn analysis"""
    st.subheader("Customers Most Likely to Cancel")
    st.caption(
        "These are **current, active customers** predicted to churn soon — not customers who have already left. "
        "Churn Risk is the model's probability estimate. Revenue Impact Score (0–100) weights that risk "
        "by monthly revenue, so a high-value customer with moderate risk can outrank a low-value customer with high risk."
    )

    at_risk = st.session_state.at_risk_customers

    # ── Fallback notice if threshold wasn't met ────────────────────────────
    threshold_pct = HIGH_RISK_THRESHOLD * 100
    max_prob = float(at_risk['churn_probability'].max()) if not at_risk.empty else 0.0
    if max_prob < HIGH_RISK_THRESHOLD:
        st.info(
            f"ℹ️ No customers currently exceed the {threshold_pct:.0f}% churn risk threshold "
            f"(highest seen: {max_prob*100:.1f}%). "
            f"Showing the top {len(at_risk)} customers by Revenue Impact Score instead."
        )

    # Display at-risk customers table
    display_df = at_risk.head(TOP_N_AT_RISK).copy()
    display_df['churn_probability'] = (display_df['churn_probability'] * 100).round(1)
    display_df['current_revenue'] = display_df['current_revenue'].round(2)
    display_df['risk_score'] = display_df['risk_score'].round(1)

    st.dataframe(
        display_df[['customer_id', 'churn_probability', 'current_revenue', 'risk_score', 'tenure_months']],
        width='stretch',
        column_config={
            "customer_id": "Customer ID",
            "churn_probability": st.column_config.NumberColumn("Churn Risk (%)", format="%.1f%%"),
            "current_revenue": st.column_config.NumberColumn("Monthly Revenue ($)", format="$%.2f"),
            "risk_score": st.column_config.NumberColumn("Revenue Impact Score (0–100)", format="%.1f"),
            "tenure_months": st.column_config.NumberColumn("Tenure (Months)", format="%.1f")
        }
    )

    # Model performance
    st.markdown("**Churn Model Performance**")
    st.caption(
        "These scores measure how accurately the model identifies customers who actually churned in historical data. "
        "AUC above 0.75 is considered good for churn prediction."
    )
    predictor = st.session_state.churn_predictor
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("AUC", _fmt(predictor.metrics.get('auc'), '.3f'))
        st.caption("0.5 = random, 1.0 = perfect")
    with col2:
        st.metric("Precision", _fmt(predictor.metrics.get('precision'), '.3f'))
        st.caption("Of flagged churners, % who actually churned")
    with col3:
        st.metric("Recall", _fmt(predictor.metrics.get('recall'), '.3f'))
        st.caption("Of all actual churners, % the model caught")
    with col4:
        st.metric("F1 Score", _fmt(predictor.metrics.get('f1'), '.3f'))
        st.caption("Balance of Precision and Recall (higher = better)")

    # Dynamic recommendations based on actual data
    st.markdown("**💡 Recommended Actions**")
    high_risk_count = int((at_risk['churn_probability'] >= HIGH_RISK_THRESHOLD).sum())
    top5 = at_risk.head(5)
    top5_rev = top5['current_revenue'].sum() if not top5.empty else 0

    if high_risk_count > 0:
        st.markdown(
            f"- 🎯 **{high_risk_count} customer{'s' if high_risk_count != 1 else ''} exceed "
            f"{threshold_pct:.0f}% churn risk** — prioritise personal outreach immediately"
        )
    else:
        st.markdown(
            f"- 🟡 No customers currently exceed {threshold_pct:.0f}% risk — monitor the top accounts below regularly"
        )
    st.markdown(
        f"- 💰 Your top 5 at-risk accounts represent **${top5_rev:,.0f}/month** in combined revenue — "
        f"assign dedicated customer success coverage"
    )
    st.markdown(
        "- 📞 For customers with tenure < 3 months and high risk, focus on onboarding quality — "
        "early churn is usually a product-fit issue, not a price issue"
    )
    st.markdown(
        "- 📊 Export the CSV below and share with your sales/CS team to action this week"
    )


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
    st.plotly_chart(fig_mrr, width='stretch')
    
    # Churn rate over time
    fig_churn = px.line(
        metrics['churn_series'],
        x='date',
        y='churn_rate',
        title='Monthly Churn Rate (%)'
    )
    fig_churn.update_layout(height=400)
    st.plotly_chart(fig_churn, width='stretch')


def show_export_section():
    """Export and download options"""
    st.subheader("📥 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        at_risk = st.session_state.at_risk_customers
        csv = at_risk.to_csv(index=False)
        st.download_button(
            label="📄 At-Risk Customers CSV",
            data=csv,
            file_name=f"at_risk_customers_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width='stretch'
        )
        st.caption("Customer IDs, churn risk %, revenue, and tenure — share with your CS team.")

    with col2:
        forecast_summary = st.session_state.forecaster.get_forecast_summary()
        csv = forecast_summary.to_csv(index=False)
        st.download_button(
            label="📈 MRR Forecast CSV",
            data=csv,
            file_name=f"mrr_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width='stretch'
        )
        st.caption("Monthly actuals + 3-month forecast with upper/lower confidence bounds.")

    with col3:
        if st.button("📊 Generate PowerPoint Report", width='stretch'):
            with st.spinner("Generating PowerPoint report..."):
                try:
                    from src.insights_generator import generate_insights
                    from src.report_builder import build_full_report

                    metrics = st.session_state.metrics.copy()
                    metrics['forecast_mrr'] = st.session_state.future_forecast['predicted'].iloc[-1]
                    metrics['at_risk_count'] = len(st.session_state.at_risk_customers)

                    insights = generate_insights(
                        metrics,
                        st.session_state.at_risk_customers
                    )

                    filepath = build_full_report(
                        metrics,
                        st.session_state.future_forecast,
                        st.session_state.at_risk_customers,
                        insights,
                        st.session_state.forecaster
                    )

                    report_path = Path(filepath)
                    st.session_state.generated_report_bytes = report_path.read_bytes()
                    st.session_state.generated_report_name = report_path.name
                    st.success("✓ Report generated! Download it below.")

                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

        st.caption("Executive summary + charts in a ready-to-present PowerPoint file.")

        if st.session_state.get('generated_report_bytes'):
            st.download_button(
                label="📥 Download PowerPoint",
                data=st.session_state.generated_report_bytes,
                file_name=st.session_state.generated_report_name or f"sales_intelligence_report_{datetime.now().strftime('%Y%m%d')}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True
            )


def main():
    """Main app flow"""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        logo_path = Path("assets/project logo.png")
        if logo_path.exists():
            st.image(str(logo_path), width='stretch')
        st.markdown("---")

        # ── 3-step progress indicator ──────────────────────────────────────
        step1_done   = st.session_state.data_loaded
        step2_done   = st.session_state.analysis_complete

        def _step(icon, label, state):
            css = {"done": "step-done", "active": "step-active", "todo": "step-todo"}[state]
            st.markdown(f'<span class="{css}">{icon} {label}</span>', unsafe_allow_html=True)

        st.markdown("### Your Progress")
        _step("✅" if step1_done  else "1️⃣", "Upload Data",    "done" if step1_done  else "active")
        _step("✅" if step2_done  else "2️⃣", "Run Analysis",   "done" if step2_done  else ("active" if step1_done else "todo"))
        _step("3️⃣",                           "View Results & Export", "active" if step2_done else "todo")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Sales Intelligence Copilot** uses AI to:
        - Forecast revenue trends
        - Predict customer churn
        - Identify at-risk customers
        - Generate actionable insights
        """)

        if st.button("🔄 Reset / Start Over"):
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