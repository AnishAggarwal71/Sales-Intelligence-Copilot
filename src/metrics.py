"""
Business Metrics Module
Calculates key SaaS metrics: MRR, ARR, Churn, CLV, ARPU, Retention
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate SaaS business metrics"""
    
    def __init__(self, subscriptions_df: pd.DataFrame, customers_df: pd.DataFrame = None):
        """
        Initialize with subscription and customer data
        
        Args:
            subscriptions_df: DataFrame with subscription records
            customers_df: Optional DataFrame with customer details
        """
        self.subscriptions = subscriptions_df.copy()
        self.customers = customers_df.copy() if customers_df is not None else None
        
        # Ensure date columns are datetime
        if 'period_start' in self.subscriptions.columns:
            self.subscriptions['period_start'] = pd.to_datetime(
                self.subscriptions['period_start']
            )
        if 'period_end' in self.subscriptions.columns:
            self.subscriptions['period_end'] = pd.to_datetime(
                self.subscriptions['period_end']
            )
    
    def calculate_mrr(self, as_of_date: datetime = None) -> pd.DataFrame:
        """
        Calculate Monthly Recurring Revenue over time
        
        Args:
            as_of_date: Calculate MRR as of this date (default: latest date)
            
        Returns:
            DataFrame with date and MRR
        """
        logger.info("Calculating MRR time series...")
        
        # Get active subscriptions for each month
        df = self.subscriptions[self.subscriptions['active'] == True].copy()
        
        # Extract year-month
        df['year_month'] = df['period_start'].dt.to_period('M')
        
        # Calculate MRR per month (sum of revenue for active subscriptions)
        mrr_series = df.groupby('year_month')['revenue'].sum().reset_index()
        mrr_series.columns = ['year_month', 'MRR']
        
        # Convert period to timestamp for plotting
        mrr_series['date'] = mrr_series['year_month'].dt.to_timestamp()
        
        logger.info(f"✓ MRR calculated for {len(mrr_series)} months")
        
        return mrr_series[['date', 'MRR']]
    
    def calculate_current_mrr(self) -> float:
        """
        Calculate current MRR (most recent month)
        
        Returns:
            Current MRR value
        """
        mrr_series = self.calculate_mrr()
        if len(mrr_series) > 0:
            return float(mrr_series.iloc[-1]['MRR'])
        return 0.0
    
    def calculate_arr(self) -> float:
        """
        Calculate Annual Recurring Revenue (MRR * 12)
        
        Returns:
            ARR value
        """
        current_mrr = self.calculate_current_mrr()
        return current_mrr * 12
    
    def calculate_churn_rate(self, period: str = 'monthly') -> pd.DataFrame:
        """
        Calculate churn rate over time
        
        Args:
            period: 'monthly' or 'quarterly'
            
        Returns:
            DataFrame with period and churn rate
        """
        logger.info(f"Calculating {period} churn rate...")
        
        df = self.subscriptions.copy()
        
        # Determine grouping period
        if period == 'monthly':
            df['period'] = df['period_start'].dt.to_period('M')
        else:  # quarterly
            df['period'] = df['period_start'].dt.to_period('Q')
        
        # Calculate customers at start and churned per period
        churn_data = []
        
        for period_val in df['period'].unique():
            period_df = df[df['period'] == period_val]
            
            # Customers at start of period (active at beginning)
            customers_start = period_df['customer_id'].nunique()
            
            # Customers who churned (churn_flag = 1)
            churned = period_df[period_df['churn_flag'] == 1]['customer_id'].nunique()
            
            # Churn rate
            churn_rate = (churned / customers_start * 100) if customers_start > 0 else 0
            
            churn_data.append({
                'period': period_val,
                'customers_start': customers_start,
                'churned': churned,
                'churn_rate': churn_rate
            })
        
        churn_df = pd.DataFrame(churn_data)
        churn_df['date'] = churn_df['period'].dt.to_timestamp()
        churn_df = churn_df.sort_values('date')
        
        logger.info(f"✓ Churn rate calculated for {len(churn_df)} periods")
        
        return churn_df
    
    def calculate_current_churn_rate(self) -> float:
        """
        Calculate current month's churn rate
        
        Returns:
            Churn rate as percentage
        """
        churn_series = self.calculate_churn_rate()
        if len(churn_series) > 0:
            return float(churn_series.iloc[-1]['churn_rate'])
        return 0.0
    
    def calculate_retention_rate(self) -> pd.DataFrame:
        """
        Calculate customer retention rate (1 - churn rate)
        
        Returns:
            DataFrame with retention rate over time
        """
        churn_df = self.calculate_churn_rate()
        churn_df['retention_rate'] = 100 - churn_df['churn_rate']
        return churn_df[['date', 'retention_rate', 'churn_rate']]
    
    def calculate_arpu(self) -> pd.DataFrame:
        """
        Calculate Average Revenue Per User over time
        
        Returns:
            DataFrame with date and ARPU
        """
        logger.info("Calculating ARPU...")
        
        df = self.subscriptions[self.subscriptions['active'] == True].copy()
        df['year_month'] = df['period_start'].dt.to_period('M')
        
        # Calculate ARPU per month
        arpu_data = df.groupby('year_month').agg({
            'revenue': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        arpu_data['ARPU'] = arpu_data['revenue'] / arpu_data['customer_id']
        arpu_data['date'] = arpu_data['year_month'].dt.to_timestamp()
        
        logger.info(f"✓ ARPU calculated for {len(arpu_data)} months")
        
        return arpu_data[['date', 'ARPU']]
    
    def calculate_current_arpu(self) -> float:
        """Get current ARPU"""
        arpu_series = self.calculate_arpu()
        if len(arpu_series) > 0:
            return float(arpu_series.iloc[-1]['ARPU'])
        return 0.0
    
    def calculate_clv(self, method: str = 'simple') -> float:
        """
        Calculate Customer Lifetime Value
        
        Args:
            method: 'simple' (ARPU / churn_rate) or 'cohort' (actual historical)
            
        Returns:
            CLV value
        """
        if method == 'simple':
            arpu = self.calculate_current_arpu()
            churn_rate = self.calculate_current_churn_rate() / 100  # Convert to decimal
            
            if churn_rate > 0:
                clv = arpu / churn_rate
            else:
                clv = arpu * 36  # Assume 36 month lifetime if no churn
            
            logger.info(f"✓ CLV (simple): ${clv:,.2f}")
            return clv
        
        else:  # cohort method
            # Calculate actual average lifetime value per customer
            customer_ltv = self.subscriptions.groupby('customer_id')['revenue'].sum()
            clv = customer_ltv.mean()
            
            logger.info(f"✓ CLV (cohort): ${clv:,.2f}")
            return clv
    
    def calculate_active_customers(self) -> int:
        """
        Get count of currently active customers
        
        Returns:
            Number of active customers
        """
        active = self.subscriptions[
            self.subscriptions['active'] == True
        ]['customer_id'].nunique()
        
        return int(active)
    
    def calculate_customer_cohorts(self) -> pd.DataFrame:
        """
        Calculate cohort retention analysis
        
        Returns:
            DataFrame with cohort retention rates
        """
        logger.info("Calculating cohort retention...")
        
        if self.customers is None:
            logger.warning("Customer data not available for cohort analysis")
            return pd.DataFrame()
        
        # Merge to get signup dates
        df = self.subscriptions.merge(
            self.customers[['customer_id', 'signup_date']],
            on='customer_id',
            how='left'
        )
        
        # Define cohorts by signup month
        df['cohort'] = pd.to_datetime(df['signup_date']).dt.to_period('M')
        df['period'] = df['period_start'].dt.to_period('M')
        
        # Calculate months since signup
        df['months_since_signup'] = (
            (df['period'] - df['cohort']).apply(lambda x: x.n)
        )
        
        # Calculate retention by cohort and month
        cohort_data = df.groupby(['cohort', 'months_since_signup']).agg({
            'customer_id': 'nunique',
            'active': 'sum'
        }).reset_index()
        
        # Calculate retention rate
        cohort_sizes = df.groupby('cohort')['customer_id'].nunique()
        cohort_data['cohort_size'] = cohort_data['cohort'].map(cohort_sizes)
        cohort_data['retention_rate'] = (
            cohort_data['active'] / cohort_data['cohort_size'] * 100
        )
        
        logger.info(f"✓ Cohort analysis complete for {cohort_data['cohort'].nunique()} cohorts")
        
        return cohort_data
    
    def calculate_growth_metrics(self) -> Dict:
        """
        Calculate growth metrics (MoM, QoQ)
        
        Returns:
            Dictionary with growth metrics
        """
        mrr_series = self.calculate_mrr()
        
        if len(mrr_series) < 2:
            return {'mrr_growth_mom': 0, 'mrr_growth_qoq': 0}
        
        # Month-over-Month growth
        current_mrr = mrr_series.iloc[-1]['MRR']
        previous_mrr = mrr_series.iloc[-2]['MRR']
        mrr_growth_mom = ((current_mrr - previous_mrr) / previous_mrr * 100) if previous_mrr > 0 else 0
        
        # Quarter-over-Quarter growth (if enough data)
        if len(mrr_series) >= 4:
            current_quarter = mrr_series.iloc[-3:]['MRR'].mean()
            previous_quarter = mrr_series.iloc[-6:-3]['MRR'].mean()
            mrr_growth_qoq = ((current_quarter - previous_quarter) / previous_quarter * 100) if previous_quarter > 0 else 0
        else:
            mrr_growth_qoq = 0
        
        return {
            'mrr_growth_mom': round(mrr_growth_mom, 2),
            'mrr_growth_qoq': round(mrr_growth_qoq, 2)
        }
    
    def get_dashboard_metrics(self) -> Dict:
        """
        Get all key metrics for dashboard display
        
        Returns:
            Dictionary with all important metrics
        """
        logger.info("=" * 60)
        logger.info("CALCULATING DASHBOARD METRICS")
        logger.info("=" * 60)
        
        growth = self.calculate_growth_metrics()
        
        metrics = {
            # Revenue metrics
            'current_mrr': self.calculate_current_mrr(),
            'arr': self.calculate_arr(),
            'mrr_growth_mom': growth['mrr_growth_mom'],
            
            # Customer metrics
            'active_customers': self.calculate_active_customers(),
            'arpu': self.calculate_current_arpu(),
            'clv': self.calculate_clv(method='simple'),
            
            # Churn & retention
            'churn_rate': self.calculate_current_churn_rate(),
            'retention_rate': 100 - self.calculate_current_churn_rate(),
            
            # Time series data
            'mrr_series': self.calculate_mrr(),
            'churn_series': self.calculate_churn_rate(),
            'arpu_series': self.calculate_arpu()
        }
        
        logger.info(f"✓ MRR: ${metrics['current_mrr']:,.2f}")
        logger.info(f"✓ Active Customers: {metrics['active_customers']:,}")
        logger.info(f"✓ Churn Rate: {metrics['churn_rate']:.2f}%")
        logger.info(f"✓ ARPU: ${metrics['arpu']:,.2f}")
        logger.info(f"✓ CLV: ${metrics['clv']:,.2f}")
        
        return metrics


# Convenience functions
def calculate_all_metrics(subscriptions_df: pd.DataFrame, 
                         customers_df: pd.DataFrame = None) -> Dict:
    """
    Quick function to calculate all metrics
    
    Args:
        subscriptions_df: Subscriptions DataFrame
        customers_df: Optional customers DataFrame
        
    Returns:
        Dictionary with all metrics
    """
    calculator = MetricsCalculator(subscriptions_df, customers_df)
    return calculator.get_dashboard_metrics()


# Test the module
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import SAMPLE_DATA_DIR
    
    # Load sample data
    subscriptions_file = SAMPLE_DATA_DIR / "subscriptions.csv"
    customers_file = SAMPLE_DATA_DIR / "customers.csv"
    
    if subscriptions_file.exists():
        print("Loading data...")
        subscriptions = pd.read_csv(subscriptions_file)
        customers = pd.read_csv(customers_file) if customers_file.exists() else None
        
        # Calculate metrics
        calculator = MetricsCalculator(subscriptions, customers)
        metrics = calculator.get_dashboard_metrics()
        
        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)
        print(f"Current MRR: ${metrics['current_mrr']:,.2f}")
        print(f"ARR: ${metrics['arr']:,.2f}")
        print(f"MRR Growth (MoM): {metrics['mrr_growth_mom']:+.2f}%")
        print(f"Active Customers: {metrics['active_customers']:,}")
        print(f"ARPU: ${metrics['arpu']:,.2f}")
        print(f"CLV: ${metrics['clv']:,.2f}")
        print(f"Churn Rate: {metrics['churn_rate']:.2f}%")
        print(f"Retention Rate: {metrics['retention_rate']:.2f}%")
        
        print("\n✓ Metrics calculation test successful!")
    else:
        print("Sample data not found. Run generate_data.py first.")