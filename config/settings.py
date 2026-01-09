"""
Configuration settings for Sales Intelligence Copilot
Central location for all constants and configuration parameters
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA SCHEMA DEFINITIONS
# ============================================================================

# Expected columns for each data file
CUSTOMERS_SCHEMA = {
    'customer_id': 'string',
    'signup_date': 'datetime',
    'country': 'string',
    'plan': 'string',  # basic, pro, enterprise
    'monthly_price': 'float',
    'segment': 'string',  # SMB, Mid, Enterprise
    'first_source': 'string'
}

SUBSCRIPTIONS_SCHEMA = {
    'subscription_id': 'string',
    'customer_id': 'string',
    'period_start': 'datetime',
    'period_end': 'datetime',
    'revenue': 'float',
    'is_renewal': 'bool',
    'churn_flag': 'int',  # 0 or 1
    'active': 'bool',
    'num_logins': 'int',
    'feature_x_usage': 'float'
}

TRANSACTIONS_SCHEMA = {
    'transaction_id': 'string',
    'customer_id': 'string',
    'amount': 'float',
    'transaction_date': 'datetime',
    'payment_method': 'string'
}

# ============================================================================
# BUSINESS METRICS CONFIGURATION
# ============================================================================

# Churn definition
CHURN_WINDOW_DAYS = 30  # Predict churn in next 30 days
INACTIVE_THRESHOLD_DAYS = 60  # Consider churned if inactive for this long

# Feature engineering windows
RECENCY_WINDOWS = [30, 90, 180]  # Days for recency calculations
FREQUENCY_WINDOWS = [30, 90, 180]  # Days for frequency calculations
MONETARY_WINDOWS = [30, 90, 180]  # Days for monetary calculations

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.7  # Churn probability threshold for high risk
TOP_N_AT_RISK = 20  # Number of at-risk customers to highlight

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Forecasting settings
FORECAST_HORIZON_DAYS = 90  # Forecast next 90 days
FORECAST_FREQUENCY = 'D'  # Daily frequency
PROPHET_SEASONALITY_MODE = 'multiplicative'
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

# Churn model settings
CHURN_TEST_SIZE = 0.2
CHURN_RANDOM_STATE = 42
CHURN_CLASS_WEIGHT = 'balanced'  # Handle class imbalance

# Model evaluation
MIN_AUC_THRESHOLD = 0.75  # Minimum acceptable AUC for churn model
MIN_MAPE_THRESHOLD = 20.0  # Maximum acceptable MAPE for forecast (%)

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# OpenAI settings
OPENAI_MODEL = "gpt-4o-mini"  # Cost-efficient model
OPENAI_MAX_TOKENS = 800  # Limit for cost control
OPENAI_TEMPERATURE = 0.3  # Lower for more consistent outputs

# Prompt templates
EXECUTIVE_SUMMARY_PROMPT = """You are a business analyst creating an executive summary for a SaaS company.

Based on the following metrics, write a concise 3-bullet executive summary (2-3 sentences per bullet):

Current MRR: ${current_mrr:,.0f}
MRR Growth (MoM): {mrr_growth:+.1f}%
Forecasted MRR (90 days): ${forecast_mrr:,.0f}
Current Churn Rate: {churn_rate:.1f}%
Active Customers: {active_customers:,}
At-Risk High-Value Customers: {at_risk_count}

Focus on:
1. Revenue trajectory and forecast
2. Customer health and churn risk
3. Key business implications

Keep it executive-level, actionable, and data-driven."""

RECOMMENDATIONS_PROMPT = """You are a strategic business advisor for a SaaS company.

Based on this analysis:
- Top at-risk customers: {at_risk_summary}
- Churn rate: {churn_rate:.1f}%
- MRR trend: {mrr_trend}

Provide 3 specific, actionable recommendations to:
1. Reduce churn
2. Protect revenue from at-risk customers
3. Improve customer retention

Each recommendation should be 2-3 sentences with clear next steps."""

# ============================================================================
# REPORTING CONFIGURATION
# ============================================================================

# PPT settings
PPT_TITLE = "Sales Intelligence Report"
PPT_SUBTITLE = "Revenue Forecast & Churn Analysis"
CHART_DPI = 150  # Resolution for embedded charts
CHART_WIDTH_INCHES = 8
CHART_HEIGHT_INCHES = 5

# Chart styling
PLOTLY_THEME = "plotly_white"
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'neutral': '#7f7f7f'
}

# ============================================================================
# SYNTHETIC DATA GENERATION SETTINGS
# ============================================================================

SYNTHETIC_DATA_CONFIG = {
    'n_customers': 10000,
    'start_date': '2021-01-01',
    'end_date': '2024-12-31',
    'base_churn_rate': 0.05,  # 5% monthly churn
    'seasonality_amplitude': 0.15,  # 15% seasonal variation
    'growth_rate': 0.02,  # 2% monthly growth
    'plans': {
        'basic': {'price': 29.0, 'weight': 0.5},
        'pro': {'price': 99.0, 'weight': 0.35},
        'enterprise': {'price': 499.0, 'weight': 0.15}
    },
    'segments': {
        'SMB': 0.6,
        'Mid': 0.3,
        'Enterprise': 0.1
    }
}

# ============================================================================
# VALIDATION RULES
# ============================================================================

DATA_QUALITY_RULES = {
    'max_missing_percent': 0.2,  # Fail if >20% missing in critical columns
    'critical_columns': ['customer_id', 'revenue', 'period_start'],
    'date_range_check': True,
    'duplicate_check': True
}

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

def get_openai_api_key():
    """Retrieve OpenAI API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in .env file or environment."
        )
    return api_key

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'