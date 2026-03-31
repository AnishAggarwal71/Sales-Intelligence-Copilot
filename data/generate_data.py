"""
Synthetic Data Generator for Sales Intelligence Copilot
Generates realistic SaaS subscription data for testing and demos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import SYNTHETIC_DATA_CONFIG

# Configuration - keep a clean 24-month history for Prophet forecasting
N_CUSTOMERS = SYNTHETIC_DATA_CONFIG.get('n_customers', 4500)
START_DATE = datetime.fromisoformat(SYNTHETIC_DATA_CONFIG.get('start_date', '2024-01-01'))
END_DATE = datetime.fromisoformat(SYNTHETIC_DATA_CONFIG.get('end_date', '2025-12-31'))
OUTPUT_DIR = Path(__file__).parent / "sample"
BASE_CHURN_RATE = SYNTHETIC_DATA_CONFIG.get('base_churn_rate', 0.028)
MONTHLY_GROWTH_RATE = SYNTHETIC_DATA_CONFIG.get('growth_rate', 0.035)
SEASONALITY_AMPLITUDE = SYNTHETIC_DATA_CONFIG.get('seasonality_amplitude', 0.10)

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def get_seasonality_factor(month: int) -> float:
    """Return a realistic seasonal multiplier with summer and year-end dips."""
    seasonal_profile = {
        1: -0.04,
        2: -0.01,
        3: 0.02,
        4: 0.05,
        5: 0.07,
        6: 0.03,
        7: -0.06,
        8: -0.12,
        9: -0.02,
        10: 0.04,
        11: 0.06,
        12: -0.08,
    }
    return 1 + seasonal_profile.get(month, 0.0) * (SEASONALITY_AMPLITUDE / 0.10)


def generate_customers(n_customers):
    """Generate customer master data with a realistic installed base plus 24 months of growth."""
    print(f"Generating {n_customers:,} customers across 24 months...")

    countries = ['US', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'India', 'Singapore']
    country_weights = [0.42, 0.14, 0.09, 0.1, 0.07, 0.07, 0.07, 0.04]

    sources = ['organic', 'paid_search', 'referral', 'direct', 'content', 'partner']
    source_weights = [0.28, 0.18, 0.14, 0.14, 0.16, 0.10]

    plans_config = SYNTHETIC_DATA_CONFIG.get('plans', {
        'basic': {'price': 29.0, 'weight': 0.55},
        'pro': {'price': 99.0, 'weight': 0.33},
        'enterprise': {'price': 499.0, 'weight': 0.12}
    })
    plans = list(plans_config.keys())
    plan_prices = {plan: details['price'] for plan, details in plans_config.items()}
    base_plan_weights = {plan: details['weight'] for plan, details in plans_config.items()}

    segment_mix = {
        'basic': {'SMB': 0.84, 'Mid': 0.14, 'Enterprise': 0.02},
        'pro': {'SMB': 0.38, 'Mid': 0.50, 'Enterprise': 0.12},
        'enterprise': {'SMB': 0.05, 'Mid': 0.35, 'Enterprise': 0.60},
    }
    segment_price_multiplier = {'SMB': 0.98, 'Mid': 1.05, 'Enterprise': 1.12}

    def build_customer_row(customer_number: int, signup_date, maturity: float):
        plan_weights = {
            'basic': max(0.45, base_plan_weights.get('basic', 0.55) - 0.10 * maturity),
            'pro': min(0.40, base_plan_weights.get('pro', 0.33) + 0.06 * maturity),
            'enterprise': min(0.18, base_plan_weights.get('enterprise', 0.12) + 0.04 * maturity),
        }
        plan_probability = np.array([plan_weights[plan] for plan in plans], dtype=float)
        plan_probability /= plan_probability.sum()
        plan = np.random.choice(plans, p=plan_probability)

        segment_names = list(segment_mix[plan].keys())
        segment_probability = list(segment_mix[plan].values())
        segment = np.random.choice(segment_names, p=segment_probability)

        price_growth_uplift = 1 + (0.04 * max(maturity, 0))
        monthly_price = (
            plan_prices[plan]
            * segment_price_multiplier[segment]
            * price_growth_uplift
            * np.random.uniform(0.93, 1.08)
        )

        return {
            'customer_id': f'CUST_{customer_number:06d}',
            'signup_date': pd.Timestamp(signup_date).to_pydatetime(),
            'country': np.random.choice(countries, p=country_weights),
            'plan': plan,
            'segment': segment,
            'first_source': np.random.choice(sources, p=source_weights),
            'monthly_price': round(monthly_price, 2)
        }

    months = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    growth_curve = np.array([(1 + MONTHLY_GROWTH_RATE) ** idx for idx in range(len(months))])
    seasonality_curve = np.array([get_seasonality_factor(month.month) for month in months])
    month_weights = growth_curve * seasonality_curve

    existing_base_count = max(250, int(n_customers * 0.18))
    net_new_customers = n_customers - existing_base_count
    monthly_customer_counts = np.random.multinomial(net_new_customers, month_weights / month_weights.sum())

    customer_rows = []
    customer_id = 1
    base_start = pd.Timestamp(START_DATE) - pd.DateOffset(months=9)

    for _ in range(existing_base_count):
        signup_month = base_start + pd.DateOffset(months=int(np.random.randint(0, 9)))
        days_in_month = int((signup_month + pd.offsets.MonthEnd(1)).day)
        signup_offset = int(np.random.randint(0, days_in_month))
        signup_date = signup_month + timedelta(days=signup_offset)
        customer_rows.append(build_customer_row(customer_id, signup_date, maturity=0.15))
        customer_id += 1

    for idx, (month_start, customer_count) in enumerate(zip(months, monthly_customer_counts)):
        maturity = idx / max(len(months) - 1, 1)
        days_in_month = int((month_start + pd.offsets.MonthEnd(1)).day)

        for _ in range(customer_count):
            signup_offset = int(np.random.randint(0, days_in_month))
            signup_date = month_start + timedelta(days=signup_offset)
            customer_rows.append(build_customer_row(customer_id, signup_date, maturity=maturity))
            customer_id += 1

    customers = pd.DataFrame(customer_rows)
    return customers.sort_values('signup_date').reset_index(drop=True)


def generate_subscriptions(customers_df):
    """Generate 24 months of subscriptions with growth, churn, upsell, and seasonal dips."""
    print("Generating subscription history with realistic growth and seasonality...")

    subscriptions = []
    subscription_id = 1
    final_period = pd.Timestamp(END_DATE).to_period('M').to_timestamp()

    for _, customer in customers_df.sort_values('signup_date').iterrows():
        customer_id = customer['customer_id']
        signup_date = pd.Timestamp(customer['signup_date'])
        current_period = max(
            signup_date.to_period('M').to_timestamp(),
            pd.Timestamp(START_DATE).to_period('M').to_timestamp()
        )
        monthly_price = float(customer['monthly_price'])
        plan = customer['plan']
        segment = customer['segment']

        churn_multiplier = {'basic': 1.30, 'pro': 1.00, 'enterprise': 0.65}[plan]
        segment_churn_adjustment = {'SMB': 1.10, 'Mid': 0.95, 'Enterprise': 0.80}[segment]
        base_engagement = {'basic': 0.42, 'pro': 0.58, 'enterprise': 0.74}[plan]

        months_since_signup = 0
        is_active = True
        expansion_multiplier = 1.0

        while current_period <= final_period and is_active:
            period_end = current_period + pd.offsets.MonthEnd(1)
            months_since_signup += 1

            engagement_level = np.clip(
                np.random.normal(base_engagement + min(months_since_signup, 12) * 0.01, 0.12),
                0.05,
                0.98
            )
            num_logins = int(max(0, np.random.poisson(18 + 42 * engagement_level + months_since_signup * 0.6)))
            feature_usage = round(np.clip(np.random.normal(45 + 45 * engagement_level, 14), 0, 100), 2)

            period_churn_rate = BASE_CHURN_RATE * churn_multiplier * segment_churn_adjustment
            if months_since_signup <= 3:
                period_churn_rate *= 1.45
            elif months_since_signup >= 12:
                period_churn_rate *= 0.85

            if engagement_level < 0.25:
                period_churn_rate *= 1.80
            elif engagement_level > 0.75:
                period_churn_rate *= 0.75

            if current_period.month in (7, 8, 12):
                period_churn_rate *= 1.10

            if months_since_signup >= 6 and engagement_level > 0.55 and np.random.random() < 0.12:
                expansion_multiplier *= np.random.uniform(1.01, 1.04)
            if months_since_signup % 12 == 0 and np.random.random() < 0.65:
                expansion_multiplier *= np.random.uniform(1.02, 1.06)

            revenue = (
                monthly_price
                * expansion_multiplier
                * get_seasonality_factor(current_period.month)
                * np.random.uniform(0.98, 1.03)
            )

            if months_since_signup == 1:
                active_days = max(1, int(period_end.day - signup_date.day + 1))
                revenue *= active_days / period_end.day

            churned = np.random.random() < min(period_churn_rate, 0.35)

            subscriptions.append({
                'subscription_id': f'SUB_{subscription_id:08d}',
                'customer_id': customer_id,
                'period_start': current_period,
                'period_end': period_end,
                'revenue': round(revenue, 2),
                'is_renewal': months_since_signup > 1,
                'churn_flag': 1 if churned else 0,
                'active': not churned,
                'num_logins': num_logins,
                'feature_x_usage': feature_usage
            })

            subscription_id += 1
            current_period = current_period + pd.offsets.MonthBegin(1)

            if churned:
                is_active = False

    return pd.DataFrame(subscriptions)


def generate_transactions(subscriptions_df):
    """Generate transaction records from subscriptions."""
    print("Generating transactions...")

    payment_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer', 'stripe']
    method_weights = [0.48, 0.18, 0.14, 0.08, 0.12]

    transactions = []

    for _, sub in subscriptions_df.iterrows():
        if np.random.random() < 0.97:
            payment_date = pd.Timestamp(sub['period_start']) + timedelta(days=int(np.random.randint(0, 5)))

            transactions.append({
                'transaction_id': f'TXN_{len(transactions) + 1:08d}',
                'customer_id': sub['customer_id'],
                'amount': sub['revenue'],
                'transaction_date': payment_date,
                'payment_method': np.random.choice(payment_methods, p=method_weights)
            })

    return pd.DataFrame(transactions)


def add_data_quality_issues(df, missing_rate=0.02):
    """Introduce realistic data quality issues for testing"""
    df_copy = df.copy()
    
    # Add some missing values in non-critical columns
    non_critical = ['feature_x_usage', 'num_logins', 'first_source']
    for col in non_critical:
        if col in df_copy.columns:
            mask = np.random.random(len(df_copy)) < missing_rate
            df_copy.loc[mask, col] = np.nan
    
    return df_copy


def main():
    """Generate all synthetic datasets"""
    print("=" * 60)
    print("Sales Intelligence Copilot - Data Generator")
    print("=" * 60)
    
    # Generate customers
    customers = generate_customers(N_CUSTOMERS)
    print(f"✓ Generated {len(customers):,} customers")
    
    # Generate subscriptions
    subscriptions = generate_subscriptions(customers)
    print(f"✓ Generated {len(subscriptions):,} subscription records")
    
    # Generate transactions
    transactions = generate_transactions(subscriptions)
    print(f"✓ Generated {len(transactions):,} transactions")
    
    # Add realistic data quality issues
    subscriptions = add_data_quality_issues(subscriptions, missing_rate=0.02)
    
    # Save to CSV
    customers_path = OUTPUT_DIR / "customers.csv"
    subscriptions_path = OUTPUT_DIR / "subscriptions.csv"
    transactions_path = OUTPUT_DIR / "transactions.csv"
    
    customers.to_csv(customers_path, index=False)
    subscriptions.to_csv(subscriptions_path, index=False)
    transactions.to_csv(transactions_path, index=False)
    
    print("\n" + "=" * 60)
    print("Files saved to:")
    print(f"  • {customers_path}")
    print(f"  • {subscriptions_path}")
    print(f"  • {transactions_path}")
    print("=" * 60)
    
    # Print summary statistics
    monthly_mrr = subscriptions[subscriptions['active'] == True].copy()
    monthly_mrr['month'] = pd.to_datetime(monthly_mrr['period_start']).dt.to_period('M')
    monthly_mrr = monthly_mrr.groupby('month')['revenue'].sum().reset_index()
    monthly_mrr['month'] = monthly_mrr['month'].dt.to_timestamp()

    print("\nData Summary:")
    print(f"  Customers: {len(customers):,}")
    print(f"  Date Range: {customers['signup_date'].min().date()} to {customers['signup_date'].max().date()}")
    print(f"  Revenue Months: {len(monthly_mrr):,}")
    print(f"  Subscription Records: {len(subscriptions):,}")
    print(f"  Churn Events: {subscriptions['churn_flag'].sum():,}")
    print(f"  Total Revenue: ${subscriptions['revenue'].sum():,.2f}")
    print(f"  Starting MRR: ${monthly_mrr.iloc[0]['revenue']:,.2f}")
    print(f"  Latest MRR: ${monthly_mrr.iloc[-1]['revenue']:,.2f}")
    print(f"  Latest ARR: ${monthly_mrr.iloc[-1]['revenue'] * 12:,.2f}")
    print(f"  Transactions: {len(transactions):,}")
    
    print("\nPlan Distribution:")
    print(customers['plan'].value_counts())
    
    print("\n✓ Data generation complete!")


if __name__ == "__main__":
    main()