"""
Churn Prediction Module
Predicts customer churn risk using machine learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, classification_report, roc_curve
)
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import settings
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    CHURN_TEST_SIZE,
    CHURN_RANDOM_STATE,
    CHURN_CLASS_WEIGHT,
    TOP_N_AT_RISK,
    HIGH_RISK_THRESHOLD,
    MODELS_DIR,
    RECENCY_WINDOWS,
    FREQUENCY_WINDOWS,
    MONETARY_WINDOWS
)


class ChurnPredictor:
    """Predict customer churn using machine learning"""
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize churn predictor
        
        Args:
            model_type: 'logistic' or 'random_forest'
        """
        self.model_type = model_type
        self.pipeline = None
        self.feature_names = None
        self.metrics = {}
        self.feature_importance = None
    
    def engineer_features(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for churn prediction
        
        Args:
            merged_df: Merged dataset with subscriptions and customers
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features for churn prediction...")
        
        df = merged_df.copy()
        
        # Ensure datetime columns
        df['period_start'] = pd.to_datetime(df['period_start'])
        df['period_end'] = pd.to_datetime(df['period_end'])
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        
        # Get reference date (latest date in data)
        reference_date = df['period_end'].max()
        
        # Aggregate by customer (latest subscription record)
        customer_features = df.sort_values('period_end').groupby('customer_id').last().reset_index()
        
        # 1. RECENCY FEATURES
        customer_features['days_since_last_activity'] = (
            reference_date - customer_features['period_end']
        ).dt.days
        
        # 2. TENURE FEATURES
        customer_features['tenure_days'] = (
            reference_date - customer_features['signup_date']
        ).dt.days
        customer_features['tenure_months'] = customer_features['tenure_days'] / 30
        
        # 3. FREQUENCY FEATURES (number of subscription records per customer)
        frequency = df.groupby('customer_id').size().reset_index(name='subscription_count')
        customer_features = customer_features.merge(frequency, on='customer_id', how='left')
        
        # 4. MONETARY FEATURES
        monetary = df.groupby('customer_id')['revenue'].agg([
            'sum', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        monetary.columns = [
            'customer_id', 'total_revenue', 'avg_revenue', 
            'revenue_std', 'min_revenue', 'max_revenue'
        ]
        customer_features = customer_features.merge(monetary, on='customer_id', how='left')
        
        # 5. USAGE FEATURES
        if 'num_logins' in df.columns:
            usage = df.groupby('customer_id')['num_logins'].agg([
                'sum', 'mean', 'max'
            ]).reset_index()
            usage.columns = ['customer_id', 'total_logins', 'avg_logins', 'max_logins']
            customer_features = customer_features.merge(usage, on='customer_id', how='left')
        
        if 'feature_x_usage' in df.columns:
            feature_usage = df.groupby('customer_id')['feature_x_usage'].agg([
                'mean', 'max'
            ]).reset_index()
            feature_usage.columns = ['customer_id', 'avg_feature_usage', 'max_feature_usage']
            customer_features = customer_features.merge(feature_usage, on='customer_id', how='left')
        
        # 6. PLAN & SEGMENT FEATURES (one-hot encoding)
        if 'plan' in customer_features.columns:
            plan_dummies = pd.get_dummies(customer_features['plan'], prefix='plan')
            customer_features = pd.concat([customer_features, plan_dummies], axis=1)
        
        if 'segment' in customer_features.columns:
            segment_dummies = pd.get_dummies(customer_features['segment'], prefix='segment')
            customer_features = pd.concat([customer_features, segment_dummies], axis=1)
        
        # 7. ENGAGEMENT FEATURES
        customer_features['revenue_per_month'] = (
            customer_features['total_revenue'] / customer_features['tenure_months'].clip(lower=1)
        )
        
        # 8. CHURN LABEL (target variable)
        if 'churn_flag' in customer_features.columns:
            customer_features['churned'] = customer_features['churn_flag']
        else:
            customer_features['churned'] = (~customer_features['active']).astype(int)
        
        # Fill NaN values
        customer_features = customer_features.fillna(0)
        
        logger.info(f"✓ Created {len(customer_features.columns)} features for {len(customer_features)} customers")
        
        return customer_features
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for modeling
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            List of feature column names
        """
        # Numeric features
        numeric_features = [
            'days_since_last_activity',
            'tenure_days',
            'tenure_months',
            'subscription_count',
            'total_revenue',
            'avg_revenue',
            'revenue_std',
            'min_revenue',
            'max_revenue',
            'revenue_per_month',
            'monthly_price'
        ]
        
        # Usage features (if available)
        usage_features = [
            'total_logins', 'avg_logins', 'max_logins',
            'avg_feature_usage', 'max_feature_usage'
        ]
        
        # Categorical features (one-hot encoded)
        categorical_features = [col for col in df.columns if col.startswith(('plan_', 'segment_'))]
        
        # Combine all features
        all_features = numeric_features + usage_features + categorical_features
        
        # Keep only features that exist in the dataframe
        selected_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"✓ Selected {len(selected_features)} features for modeling")
        
        return selected_features
    
    def train(self, merged_df: pd.DataFrame, test_size: float = CHURN_TEST_SIZE) -> Dict:
        """
        Train churn prediction model
        
        Args:
            merged_df: Merged dataset
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("=" * 60)
        logger.info("TRAINING CHURN PREDICTION MODEL")
        logger.info("=" * 60)
        
        # Engineer features
        features_df = self.engineer_features(merged_df)
        
        # Select features
        self.feature_names = self.select_features(features_df)
        
        # Prepare X and y
        X = features_df[self.feature_names]
        y = features_df['churned']
        
        logger.info(f"Training data: {len(X)} samples, {len(self.feature_names)} features")
        logger.info(f"Churn rate: {y.mean()*100:.2f}%")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=CHURN_RANDOM_STATE, stratify=y
        )
        
        # Create model
        if self.model_type == 'logistic':
            model = LogisticRegression(
                class_weight=CHURN_CLASS_WEIGHT,
                random_state=CHURN_RANDOM_STATE,
                max_iter=1000
            )
        else:  # random_forest
            model = RandomForestClassifier(
                n_estimators=100,
                class_weight=CHURN_CLASS_WEIGHT,
                random_state=CHURN_RANDOM_STATE,
                max_depth=10,
                n_jobs=-1
            )
        
        # Create pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        logger.info(f"✓ Model Performance:")
        logger.info(f"  AUC: {self.metrics['auc']:.3f}")
        logger.info(f"  Precision: {self.metrics['precision']:.3f}")
        logger.info(f"  Recall: {self.metrics['recall']:.3f}")
        logger.info(f"  F1 Score: {self.metrics['f1']:.3f}")
        
        # Feature importance (for tree-based models)
        if self.model_type == 'random_forest':
            importance = self.pipeline.named_steps['model'].feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return self.metrics
    
    def predict_churn_probability(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn probability for all customers
        
        Args:
            merged_df: Merged dataset
            
        Returns:
            DataFrame with customer_id and churn probability
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Predicting churn probabilities...")
        
        # Engineer features
        features_df = self.engineer_features(merged_df)
        
        # Prepare features
        X = features_df[self.feature_names]
        
        # Predict probabilities
        churn_proba = self.pipeline.predict_proba(X)[:, 1]
        
        # Create results dataframe
        results = pd.DataFrame({
            'customer_id': features_df['customer_id'],
            'churn_probability': churn_proba,
            'current_revenue': features_df.get('monthly_price', features_df.get('avg_revenue', 0)),
            'tenure_months': features_df['tenure_months'],
            'active': features_df.get('active', True)
        })
        
        # Calculate risk score (probability * revenue)
        results['risk_score'] = results['churn_probability'] * results['current_revenue']
        
        # Sort by risk score
        results = results.sort_values('risk_score', ascending=False)
        
        logger.info(f"✓ Predicted churn for {len(results)} customers")
        
        return results
    
    def get_at_risk_customers(self, merged_df: pd.DataFrame, 
                             top_n: int = TOP_N_AT_RISK,
                             min_probability: float = HIGH_RISK_THRESHOLD) -> pd.DataFrame:
        """
        Get list of high-risk customers
        
        Args:
            merged_df: Merged dataset
            top_n: Number of top at-risk customers
            min_probability: Minimum churn probability threshold
            
        Returns:
            DataFrame with top at-risk customers
        """
        predictions = self.predict_churn_probability(merged_df)
        
        # Filter by probability threshold
        at_risk = predictions[predictions['churn_probability'] >= min_probability]
        
        # Get top N by risk score
        top_at_risk = at_risk.head(top_n)
        
        logger.info(f"✓ Identified {len(at_risk)} at-risk customers (>{min_probability*100}% probability)")
        logger.info(f"✓ Top {len(top_at_risk)} by risk score")
        
        return top_at_risk
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
            
        Returns:
            Path where model was saved
        """
        if self.pipeline is None:
            raise ValueError("No model to save. Train the model first.")
        
        if filepath is None:
            filepath = MODELS_DIR / "churn_model.pkl"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✓ Model saved to {filepath}")
        
        return str(filepath)
    
    @classmethod
    def load_model(cls, filepath: str = None) -> 'ChurnPredictor':
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to model file
            
        Returns:
            ChurnPredictor instance with loaded model
        """
        if filepath is None:
            filepath = MODELS_DIR / "churn_model.pkl"
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.pipeline = model_data['pipeline']
        predictor.feature_names = model_data['feature_names']
        predictor.metrics = model_data.get('metrics', {})
        predictor.feature_importance = model_data.get('feature_importance')
        
        logger.info(f"✓ Model loaded from {filepath}")
        
        return predictor


def train_and_predict_churn(merged_df: pd.DataFrame,
                            model_type: str = 'logistic',
                            save_model: bool = True) -> Tuple[ChurnPredictor, pd.DataFrame]:
    """
    Convenience function to train model and get at-risk customers
    
    Args:
        merged_df: Merged dataset
        model_type: 'logistic' or 'random_forest'
        save_model: Whether to save the trained model
        
    Returns:
        Tuple of (predictor instance, at-risk customers DataFrame)
    """
    predictor = ChurnPredictor(model_type=model_type)
    
    # Train
    predictor.train(merged_df)
    
    # Get at-risk customers
    at_risk = predictor.get_at_risk_customers(merged_df)
    
    # Save model
    if save_model:
        predictor.save_model()
    
    return predictor, at_risk


# Test the module
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import SAMPLE_DATA_DIR
    from src.data_processing import DataProcessor
    
    # Load sample data
    customers_file = SAMPLE_DATA_DIR / "customers.csv"
    subscriptions_file = SAMPLE_DATA_DIR / "subscriptions.csv"
    
    if customers_file.exists() and subscriptions_file.exists():
        print("Loading data...")
        processor = DataProcessor()
        processor.load_customers(str(customers_file))
        processor.load_subscriptions(str(subscriptions_file))
        merged = processor.merge_datasets()
        
        # Train and predict
        predictor, at_risk = train_and_predict_churn(
            merged,
            model_type='logistic',
            save_model=True
        )
        
        print("\n" + "=" * 60)
        print("AT-RISK CUSTOMERS")
        print("=" * 60)
        print(at_risk[['customer_id', 'churn_probability', 'current_revenue', 'risk_score']].head(10))
        
        print("\n✓ Churn prediction test successful!")
    else:
        print("Sample data not found. Run generate_data.py first.")