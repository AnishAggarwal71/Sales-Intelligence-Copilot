"""
Data Processing Module
Handles CSV/XLSX loading, validation, cleaning, and merging
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging

# Import configurations
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    CUSTOMERS_SCHEMA,
    SUBSCRIPTIONS_SCHEMA,
    TRANSACTIONS_SCHEMA,
    DATA_QUALITY_RULES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main class for data loading and processing"""
    
    def __init__(self):
        self.customers = None
        self.subscriptions = None
        self.transactions = None
        self.merged_data = None
        self.quality_report = {}
    
    def load_file(self, file_path: str, file_type: str = 'auto') -> pd.DataFrame:
        """
        Load CSV or XLSX file with automatic format detection
        
        Args:
            file_path: Path to the file
            file_type: 'csv', 'xlsx', or 'auto' for automatic detection
            
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect file type
        if file_type == 'auto':
            file_type = file_path.suffix.lower().replace('.', '')
        
        logger.info(f"Loading {file_type} file: {file_path.name}")
        
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
    
    def validate_schema(self, df: pd.DataFrame, schema: Dict, 
                       data_type: str) -> Dict:
        """
        Validate DataFrame against expected schema
        
        Args:
            df: DataFrame to validate
            schema: Expected schema dictionary
            data_type: Type of data (for reporting)
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {data_type} schema...")
        
        validation = {
            'data_type': data_type,
            'valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': [],
            'warnings': []
        }
        
        # Check for missing required columns
        expected_cols = set(schema.keys())
        actual_cols = set(df.columns)
        
        validation['missing_columns'] = list(expected_cols - actual_cols)
        validation['extra_columns'] = list(actual_cols - expected_cols)
        
        if validation['missing_columns']:
            validation['valid'] = False
            logger.warning(f"Missing columns: {validation['missing_columns']}")
        
        if validation['extra_columns']:
            validation['warnings'].append(
                f"Extra columns found: {validation['extra_columns']}"
            )
        
        return validation
    
    def clean_data(self, df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
        """
        Clean and standardize data
        
        Args:
            df: DataFrame to clean
            schema: Schema with type information
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        logger.info("Cleaning data...")
        
        # Convert data types
        for col, dtype in schema.items():
            if col not in df_clean.columns:
                continue
            
            try:
                if dtype == 'datetime':
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                elif dtype == 'float':
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                elif dtype == 'int':
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    df_clean[col] = df_clean[col].fillna(0).astype(int)
                elif dtype == 'bool':
                    df_clean[col] = df_clean[col].astype(bool)
                elif dtype == 'string':
                    df_clean[col] = df_clean[col].astype(str)
                    # Clean string columns
                    df_clean[col] = df_clean[col].str.strip()
                    df_clean[col] = df_clean[col].replace('nan', np.nan)
                    
            except Exception as e:
                logger.warning(f"Could not convert {col} to {dtype}: {e}")
        
        # Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed:,} duplicate rows")
        
        return df_clean
    
    def generate_quality_report(self, df: pd.DataFrame, 
                               data_type: str) -> Dict:
        """
        Generate data quality report
        
        Args:
            df: DataFrame to analyze
            data_type: Type of data
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'data_type': data_type,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'missing_percentage': {},
            'data_quality_score': 0
        }
        
        # Calculate missing values per column
        for col in df.columns:
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            
            if missing > 0:
                report['missing_values'][col] = missing
                report['missing_percentage'][col] = round(missing_pct, 2)
        
        # Calculate overall quality score (100 - avg missing %)
        if report['missing_percentage']:
            avg_missing = np.mean(list(report['missing_percentage'].values()))
            report['data_quality_score'] = round(100 - avg_missing, 1)
        else:
            report['data_quality_score'] = 100.0
        
        return report
    
    def load_customers(self, file_path: str) -> pd.DataFrame:
        """Load and process customer data"""
        logger.info("=" * 60)
        logger.info("LOADING CUSTOMERS DATA")
        logger.info("=" * 60)
        
        df = self.load_file(file_path)
        validation = self.validate_schema(df, CUSTOMERS_SCHEMA, 'customers')
        
        if not validation['valid']:
            raise ValueError(f"Schema validation failed: {validation}")
        
        self.customers = self.clean_data(df, CUSTOMERS_SCHEMA)
        self.quality_report['customers'] = self.generate_quality_report(
            self.customers, 'customers'
        )
        
        logger.info(f"✓ Customers loaded: {len(self.customers):,} records")
        logger.info(f"✓ Quality score: {self.quality_report['customers']['data_quality_score']}%")
        
        return self.customers
    
    def load_subscriptions(self, file_path: str) -> pd.DataFrame:
        """Load and process subscription data"""
        logger.info("=" * 60)
        logger.info("LOADING SUBSCRIPTIONS DATA")
        logger.info("=" * 60)
        
        df = self.load_file(file_path)
        validation = self.validate_schema(df, SUBSCRIPTIONS_SCHEMA, 'subscriptions')
        
        if not validation['valid']:
            raise ValueError(f"Schema validation failed: {validation}")
        
        self.subscriptions = self.clean_data(df, SUBSCRIPTIONS_SCHEMA)
        self.quality_report['subscriptions'] = self.generate_quality_report(
            self.subscriptions, 'subscriptions'
        )
        
        logger.info(f"✓ Subscriptions loaded: {len(self.subscriptions):,} records")
        logger.info(f"✓ Quality score: {self.quality_report['subscriptions']['data_quality_score']}%")
        
        return self.subscriptions
    
    def load_transactions(self, file_path: str) -> pd.DataFrame:
        """Load and process transaction data"""
        logger.info("=" * 60)
        logger.info("LOADING TRANSACTIONS DATA")
        logger.info("=" * 60)
        
        df = self.load_file(file_path)
        validation = self.validate_schema(df, TRANSACTIONS_SCHEMA, 'transactions')
        
        if not validation['valid']:
            raise ValueError(f"Schema validation failed: {validation}")
        
        self.transactions = self.clean_data(df, TRANSACTIONS_SCHEMA)
        self.quality_report['transactions'] = self.generate_quality_report(
            self.transactions, 'transactions'
        )
        
        logger.info(f"✓ Transactions loaded: {len(self.transactions):,} records")
        logger.info(f"✓ Quality score: {self.quality_report['transactions']['data_quality_score']}%")
        
        return self.transactions
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all datasets into a single analytical dataset
        
        Returns:
            Merged DataFrame ready for analysis
        """
        if self.customers is None or self.subscriptions is None:
            raise ValueError("Must load customers and subscriptions first")
        
        logger.info("=" * 60)
        logger.info("MERGING DATASETS")
        logger.info("=" * 60)
        
        # Start with subscriptions as base
        merged = self.subscriptions.copy()
        
        # Merge with customers
        merged = merged.merge(
            self.customers,
            on='customer_id',
            how='left',
            suffixes=('', '_customer')
        )
        
        logger.info(f"✓ Merged subscriptions + customers: {len(merged):,} records")
        
        # Optionally merge with transactions (aggregate by customer)
        if self.transactions is not None:
            # Aggregate transaction data by customer
            txn_summary = self.transactions.groupby('customer_id').agg({
                'amount': ['sum', 'count', 'mean'],
                'transaction_date': ['min', 'max']
            }).reset_index()
            
            txn_summary.columns = [
                'customer_id',
                'total_transaction_amount',
                'transaction_count',
                'avg_transaction_amount',
                'first_transaction_date',
                'last_transaction_date'
            ]
            
            merged = merged.merge(
                txn_summary,
                on='customer_id',
                how='left'
            )
            
            logger.info(f"✓ Added transaction summary features")
        
        self.merged_data = merged
        
        logger.info(f"✓ Final merged dataset: {len(self.merged_data):,} rows, "
                   f"{len(self.merged_data.columns)} columns")
        
        return self.merged_data
    
    def get_summary(self) -> Dict:
        """Get a summary of loaded data"""
        summary = {
            'customers_count': len(self.customers) if self.customers is not None else 0,
            'subscriptions_count': len(self.subscriptions) if self.subscriptions is not None else 0,
            'transactions_count': len(self.transactions) if self.transactions is not None else 0,
            'quality_scores': {
                k: v['data_quality_score'] 
                for k, v in self.quality_report.items()
            }
        }
        
        if self.customers is not None:
            summary['date_range'] = {
                'start': self.customers['signup_date'].min(),
                'end': self.customers['signup_date'].max()
            }
        
        if self.subscriptions is not None:
            summary['active_subscriptions'] = self.subscriptions['active'].sum()
            summary['total_revenue'] = self.subscriptions['revenue'].sum()
        
        return summary


# Convenience function for quick loading
def load_all_data(customers_path: str, subscriptions_path: str, 
                 transactions_path: str = None) -> Tuple[DataProcessor, pd.DataFrame]:
    """
    Quick function to load all data files
    
    Args:
        customers_path: Path to customers file
        subscriptions_path: Path to subscriptions file
        transactions_path: Optional path to transactions file
        
    Returns:
        Tuple of (DataProcessor instance, merged DataFrame)
    """
    processor = DataProcessor()
    
    processor.load_customers(customers_path)
    processor.load_subscriptions(subscriptions_path)
    
    if transactions_path:
        processor.load_transactions(transactions_path)
    
    merged_data = processor.merge_datasets()
    
    return processor, merged_data


# Test the module
if __name__ == "__main__":
    from config.settings import SAMPLE_DATA_DIR
    
    # Test with sample data
    processor = DataProcessor()
    
    customers_file = SAMPLE_DATA_DIR / "customers.csv"
    subscriptions_file = SAMPLE_DATA_DIR / "subscriptions.csv"
    transactions_file = SAMPLE_DATA_DIR / "transactions.csv"
    
    if customers_file.exists() and subscriptions_file.exists():
        processor.load_customers(str(customers_file))
        processor.load_subscriptions(str(subscriptions_file))
        processor.load_transactions(str(transactions_file))
        
        merged = processor.merge_datasets()
        
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        summary = processor.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("\n✓ Data processing test successful!")
    else:
        print("Sample data not found. Run generate_data.py first.")