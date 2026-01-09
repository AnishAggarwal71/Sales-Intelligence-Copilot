"""
Revenue Forecasting Module
Uses Prophet to forecast MRR for the next 90 days
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from pathlib import Path
from typing import Dict, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import settings
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    FORECAST_HORIZON_DAYS,
    PROPHET_SEASONALITY_MODE,
    PROPHET_CHANGEPOINT_PRIOR_SCALE,
    MODELS_DIR
)


class MRRForecaster:
    """Forecast Monthly Recurring Revenue using Prophet"""
    
    def __init__(self, seasonality_mode: str = PROPHET_SEASONALITY_MODE):
        """
        Initialize forecaster
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
        """
        self.model = None
        self.seasonality_mode = seasonality_mode
        self.forecast_df = None
        self.actual_df = None
        self.metrics = {}
    
    def prepare_data(self, mrr_series: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns)
        
        Args:
            mrr_series: DataFrame with 'date' and 'MRR' columns
            
        Returns:
            DataFrame formatted for Prophet
        """
        df = mrr_series.copy()
        
        # Prophet requires specific column names
        df = df.rename(columns={'date': 'ds', 'MRR': 'y'})
        
        # Ensure datetime format
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        logger.info(f"Prepared {len(df)} data points for forecasting")
        logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
        
        return df
    
    def train(self, mrr_series: pd.DataFrame, 
              yearly_seasonality: bool = True,
              weekly_seasonality: bool = False) -> None:
        """
        Train Prophet model on historical MRR data
        
        Args:
            mrr_series: DataFrame with date and MRR columns
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns
        """
        logger.info("=" * 60)
        logger.info("TRAINING FORECASTING MODEL")
        logger.info("=" * 60)
        
        # Prepare data
        train_df = self.prepare_data(mrr_series)
        self.actual_df = train_df.copy()
        
        # Initialize Prophet model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=PROPHET_CHANGEPOINT_PRIOR_SCALE,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False
        )
        
        # Fit model
        logger.info("Fitting Prophet model...")
        self.model.fit(train_df)
        
        logger.info("✓ Model training complete")
    
    def predict(self, periods: int = 3) -> pd.DataFrame:
        """
        Generate forecast for future periods
        
        Args:
            periods: Number of months to forecast (default: 3 months = ~90 days)
            
        Returns:
            DataFrame with forecast results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Generating {periods}-month forecast...")
        
        # Create future dataframe for MONTHS, not days
        future = self.model.make_future_dataframe(periods=periods, freq='MS')  # MS = Month Start
        
        # Generate predictions
        forecast = self.model.predict(future)
        
        # Store forecast
        self.forecast_df = forecast
        
        logger.info(f"✓ Forecast generated for {periods} months")
        
        return forecast
    
    def get_forecast_summary(self) -> pd.DataFrame:
        """
        Get clean forecast summary with date, actual, predicted, and bounds
        
        Returns:
            DataFrame with forecast summary
        """
        if self.forecast_df is None:
            raise ValueError("No forecast available. Call predict() first.")
        
        # Get key columns
        summary = self.forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        summary.columns = ['date', 'predicted', 'lower_bound', 'upper_bound']
        
        # Merge with actual values
        if self.actual_df is not None:
            actual = self.actual_df[['ds', 'y']].copy()
            actual.columns = ['date', 'actual']
            summary = summary.merge(actual, on='date', how='left')
        
        # Round values
        for col in ['predicted', 'lower_bound', 'upper_bound', 'actual']:
            if col in summary.columns:
                summary[col] = summary[col].round(2)
        
        return summary
    
    def get_future_forecast(self, months_ahead: int = 3) -> pd.DataFrame:
        """
        Get only the future forecast (excluding historical data)
        
        Args:
            months_ahead: Number of months into future
            
        Returns:
            DataFrame with future predictions only
        """
        summary = self.get_forecast_summary()
        
        # Get last actual date
        last_actual_date = self.actual_df['ds'].max()
        
        # Filter for future only
        future_only = summary[summary['date'] > last_actual_date].head(months_ahead)
        
        return future_only
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate forecast accuracy metrics on historical data
        
        Returns:
            Dictionary with MAPE, MAE, RMSE
        """
        if self.forecast_df is None or self.actual_df is None:
            raise ValueError("Need both forecast and actual data for metrics")
        
        # Merge actual and predicted for comparison
        comparison = self.actual_df.merge(
            self.forecast_df[['ds', 'yhat']],
            on='ds',
            how='inner'
        )
        
        actual = comparison['y'].values
        predicted = comparison['yhat'].values
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        self.metrics = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2)
        }
        
        logger.info(f"✓ Forecast Metrics: MAE=${mae:,.2f}, RMSE=${rmse:,.2f}, MAPE={mape:.2f}%")
        
        return self.metrics
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model (default: models/forecast_model.pkl)
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        if filepath is None:
            filepath = MODELS_DIR / "forecast_model.pkl"
        else:
            filepath = Path(filepath)
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'actual_df': self.actual_df,
            'forecast_df': self.forecast_df,
            'metrics': self.metrics,
            'seasonality_mode': self.seasonality_mode
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✓ Model saved to {filepath}")
        
        return str(filepath)
    
    @classmethod
    def load_model(cls, filepath: str = None) -> 'MRRForecaster':
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to model file
            
        Returns:
            MRRForecaster instance with loaded model
        """
        if filepath is None:
            filepath = MODELS_DIR / "forecast_model.pkl"
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Create instance
        forecaster = cls(seasonality_mode=model_data['seasonality_mode'])
        forecaster.model = model_data['model']
        forecaster.actual_df = model_data['actual_df']
        forecaster.forecast_df = model_data['forecast_df']
        forecaster.metrics = model_data.get('metrics', {})
        
        logger.info(f"✓ Model loaded from {filepath}")
        
        return forecaster
    
    def plot_forecast(self, show_components: bool = False):
        """
        Generate forecast plot (requires matplotlib)
        
        Args:
            show_components: Also show trend, seasonality components
        """
        if self.model is None or self.forecast_df is None:
            raise ValueError("Model and forecast needed for plotting")
        
        import matplotlib.pyplot as plt
        
        # Plot forecast
        fig1 = self.model.plot(self.forecast_df)
        plt.title('MRR Forecast')
        plt.ylabel('MRR ($)')
        plt.xlabel('Date')
        
        # Plot components
        if show_components:
            fig2 = self.model.plot_components(self.forecast_df)
        
        return fig1


def train_and_forecast(mrr_series: pd.DataFrame, 
                      forecast_months: int = 3,
                      save_model: bool = True) -> Tuple[MRRForecaster, pd.DataFrame]:
    """
    Convenience function to train model and generate forecast
    
    Args:
        mrr_series: DataFrame with MRR time series
        forecast_months: Number of months to forecast (default: 3 = ~90 days)
        save_model: Whether to save the trained model
        
    Returns:
        Tuple of (forecaster instance, forecast DataFrame)
    """
    forecaster = MRRForecaster()
    
    # Train
    forecaster.train(mrr_series)
    
    # Predict
    forecaster.predict(periods=forecast_months)
    
    # Calculate metrics
    forecaster.calculate_metrics()
    
    # Save model
    if save_model:
        forecaster.save_model()
    
    # Get future forecast
    future_forecast = forecaster.get_future_forecast(months_ahead=forecast_months)
    
    return forecaster, future_forecast


# Test the module
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import SAMPLE_DATA_DIR
    from src.metrics import MetricsCalculator
    
    # Load sample data
    subscriptions_file = SAMPLE_DATA_DIR / "subscriptions.csv"
    
    if subscriptions_file.exists():
        print("Loading data...")
        subscriptions = pd.read_csv(subscriptions_file)
        
        # Calculate MRR
        calculator = MetricsCalculator(subscriptions)
        mrr_series = calculator.calculate_mrr()
        
        print(f"MRR data: {len(mrr_series)} months")
        print(f"Latest MRR: ${mrr_series.iloc[-1]['MRR']:,.2f}")
        
        # Train and forecast
        forecaster, future_forecast = train_and_forecast(
            mrr_series,
            forecast_months=3,  # 3 months ahead
            save_model=True
        )
        
        print("\n" + "=" * 60)
        print("FORECAST RESULTS")
        print("=" * 60)
        print(f"\nForecast Accuracy Metrics:")
        for metric, value in forecaster.metrics.items():
            print(f"  {metric}: {value}")
        
        print(f"\nNext 30 Days Forecast:")
        print(future_forecast.head(30)[['date', 'predicted', 'lower_bound', 'upper_bound']])
        
        print("\n✓ Forecasting test successful!")
    else:
        print("Sample data not found. Run generate_data.py first.")