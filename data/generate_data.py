"""
Synthetic Data Generator for Sales Intelligence Copilot
Generates realistic SaaS subscription data for testing and demos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Configuration
N_CUSTOMERS = 10000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 12, 31)
OUTPUT_DIR = Path(__file__).parent / "sample"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_customers(n_customers):
    """Generate customer master data"""
    print(f"Generating {n_customers:,} customers...")
    
    countries = ['US', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'India', 'Singapore']
    country_weights = [0.4, 0.15, 0.1, 0.1, 0.08, 0.07, 0.06, 0.04]
    
    plans = ['basic', 'pro', 'enterprise']
    plan_weights = [0.5, 0.35, 0.15]
    plan_prices = {'basic': 29.0, 'pro': 99.0, 'enterprise': 499.0}
    
    segments = ['SMB', 'Mid', 'Enterprise']
    segment_weights = [0.6, 0.3, 0.1]
    
    sources = ['organic', 'paid_search', 'referral', 'direct', 'content', 'partner']
    source_weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    
    # Generate signup dates (more recent signups = more customers)
    days_range = (END_DATE - START_DATE).days
    signup_dates = []
    for _ in range(n_customers):
        # Bias towards more recent signups
        days_ago = int(np.random.exponential(days_range / 3))
        days_ago = min(days_ago, days_range)
        signup_date = END_DATE - timedelta(days=days_ago)
        signup_dates.append(signup_date)
    
    customers = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(1, n_customers + 1)],
        'signup_date': signup_dates,
        'country': np.random.choice(countries, n_customers, p=country_weights),
        'plan': np.random.choice(plans, n_customers, p=plan_weights),
        'segment': np.random.choice(segments, n_customers, p=segment_weights),
        'first_source': np.random.choice(sources, n_customers, p=source_weights)
    })
    
    # Add monthly price based on plan
    customers['monthly_price'] = customers['plan'].map(plan_prices)
    
    # Add some price variation (discounts, custom pricing)
    customers['monthly_price'] *= np.random.uniform(0.85, 1.15, n_customers)
    customers['monthly_price'] = customers['monthly_price'].round(2)
    
    return customers


def generate_subscriptions(customers_df):
    """Generate subscription history with churn patterns"""
    print("Generating subscription history...")
    
    subscriptions = []
    subscription_id = 1
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        signup_date = customer['signup_date']
        monthly_price = customer['monthly_price']
        plan = customer['plan']
        
        # Churn probability based on plan and segment
        base_churn_rate = 0.05  # 5% monthly
        if plan == 'basic':
            churn_multiplier = 1.5
        elif plan == 'pro':
            churn_multiplier = 1.0
        else:  # enterprise
            churn_multiplier = 0.5
        
        # Generate monthly subscription records
        current_date = signup_date
        is_active = True
        months_since_signup = 0
        
        while current_date <= END_DATE and is_active:
            period_end = current_date + timedelta(days=30)
            
            # Usage patterns (higher for engaged customers)
            engagement_level = np.random.beta(2, 5)  # Skewed towards lower values
            num_logins = int(np.random.poisson(20 * engagement_level))
            feature_usage = round(np.random.uniform(0, 100) * engagement_level, 2)
            
            # Churn logic with early-stage and disengagement risk
            months_since_signup += 1
            
            # Higher churn in first 3 months and for low engagement
            if months_since_signup <= 3:
                period_churn_rate = base_churn_rate * churn_multiplier * 2.0
            else:
                period_churn_rate = base_churn_rate * churn_multiplier
            
            # Low engagement increases churn risk
            if engagement_level < 0.2:
                period_churn_rate *= 2.5
            
            # Determine if customer churns this period
            churned = np.random.random() < period_churn_rate
            
            # Revenue with some variation
            revenue = monthly_price * np.random.uniform(0.95, 1.05)
            
            subscriptions.append({
                'subscription_id': f'SUB_{subscription_id:08d}',
                'customer_id': customer_id,
                'period_start': current_date,
                'period_end': period_end,
                'revenue': round(revenue, 2),
                'is_renewal': months_since_signup > 1,
                'churn_flag': 1 if churned else 0,
                'active': not churned,
                'num_logins': num_logins,
                'feature_x_usage': feature_usage
            })
            
            subscription_id += 1
            current_date = period_end
            
            if churned:
                is_active = False
    
    return pd.DataFrame(subscriptions)


def generate_transactions(subscriptions_df):
    """Generate transaction records from subscriptions"""
    print("Generating transactions...")
    
    payment_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer', 'stripe']
    method_weights = [0.45, 0.20, 0.15, 0.10, 0.10]
    
    transactions = []
    
    for _, sub in subscriptions_df.iterrows():
        # Most subscriptions have a payment transaction
        if np.random.random() < 0.95:  # 95% payment success rate
            # Payment typically happens at period start
            payment_date = sub['period_start'] + timedelta(days=np.random.randint(0, 5))
            
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
    print("\nData Summary:")
    print(f"  Customers: {len(customers):,}")
    print(f"  Date Range: {customers['signup_date'].min().date()} to {customers['signup_date'].max().date()}")
    print(f"  Subscription Records: {len(subscriptions):,}")
    print(f"  Active Subscriptions: {subscriptions['active'].sum():,}")
    print(f"  Churned Subscriptions: {subscriptions['churn_flag'].sum():,}")
    print(f"  Total Revenue: ${subscriptions['revenue'].sum():,.2f}")
    print(f"  Transactions: {len(transactions):,}")
    
    print("\nPlan Distribution:")
    print(customers['plan'].value_counts())
    
    print("\n✓ Data generation complete!")


if __name__ == "__main__":
    main()