"""
Revenue Forecasting Module
Uses Prophet, with a robust fallback, to forecast monthly recurring revenue.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple
import logging
import warnings

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - handled gracefully at runtime
    Prophet = None

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import settings
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    FORECAST_HORIZON_MONTHS,
    PROPHET_SEASONALITY_MODE,
    PROPHET_CHANGEPOINT_PRIOR_SCALE,
    MODELS_DIR
)


class MRRForecaster:
    """Forecast Monthly Recurring Revenue using Prophet with a realistic fallback model."""

    def __init__(self, seasonality_mode: str = PROPHET_SEASONALITY_MODE):
        """Initialize forecaster state."""
        self.model = None
        self.model_type = 'untrained'
        self.seasonality_mode = seasonality_mode
        self.forecast_df = None
        self.actual_df = None
        self.train_df = None
        self.fallback_params = {}
        self.metrics = {}

    def prepare_data(self, mrr_series: pd.DataFrame) -> pd.DataFrame:
        """Prepare a complete monthly MRR series for forecasting."""
        df = mrr_series.copy()

        if {'date', 'MRR'}.issubset(df.columns):
            df = df.rename(columns={'date': 'ds', 'MRR': 'y'})
        elif not {'ds', 'y'}.issubset(df.columns):
            raise ValueError("mrr_series must contain either ['date', 'MRR'] or ['ds', 'y'] columns")

        df['ds'] = pd.to_datetime(df['ds']).dt.to_period('M').dt.to_timestamp()
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['ds', 'y'])

        df = df.groupby('ds', as_index=False)['y'].sum()
        df = df.sort_values('ds').reset_index(drop=True)

        full_range = pd.date_range(df['ds'].min(), df['ds'].max(), freq='MS')
        df = df.set_index('ds').reindex(full_range).rename_axis('ds').reset_index()
        df['y'] = df['y'].interpolate().ffill().bfill().clip(lower=0)

        logger.info(f"Prepared {len(df)} monthly data points for forecasting")
        logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

        return df

    def _build_training_frame(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Create a smoothed training frame to reduce noise and cap runaway forecasts."""
        prepared = train_df.copy()
        smooth_span = min(4, max(2, len(prepared)))
        prepared['y'] = prepared['y'].ewm(span=smooth_span, adjust=False).mean()

        last_value = float(prepared['y'].iloc[-1])
        cap = max(float(train_df['y'].max()) * 1.18, last_value * 1.22)
        prepared['cap'] = round(cap, 2)
        prepared['floor'] = 0.0

        return prepared

    def _fit_fallback_model(self, train_df: pd.DataFrame) -> Dict:
        """Fit a damped trend + seasonality model when Prophet backend is unavailable."""
        df = train_df.copy()
        smooth_span = min(5, max(2, len(df)))
        df['smoothed'] = df['y'].ewm(span=smooth_span, adjust=False).mean()

        rolling_baseline = df['smoothed'].rolling(window=3, min_periods=1, center=True).mean()
        seasonal_ratio = (df['y'] / rolling_baseline.replace(0, np.nan))
        seasonal_ratio = seasonal_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        seasonal_ratio = seasonal_ratio.clip(0.88, 1.12)

        month_factors = seasonal_ratio.groupby(df['ds'].dt.month).mean().reindex(range(1, 13), fill_value=1.0)
        month_factors = (month_factors / month_factors.mean()).clip(0.90, 1.10)

        pct_changes = df['smoothed'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        recent_growth = pct_changes.tail(min(6, len(pct_changes))).median() if not pct_changes.empty else 0.03

        if len(df) >= 3:
            trend_slope = np.polyfit(
                np.arange(len(df)),
                np.log(df['smoothed'].clip(lower=1)),
                1
            )[0]
            long_term_growth = np.expm1(trend_slope)
        else:
            long_term_growth = recent_growth

        monthly_growth = float(np.clip(0.65 * recent_growth + 0.35 * long_term_growth, 0.0, 0.08))
        residual_scale = float(np.clip((df['y'] - df['smoothed']).abs().mean() / max(df['y'].mean(), 1), 0.05, 0.15))

        return {
            'monthly_growth': monthly_growth,
            'seasonal_factors': month_factors.to_dict(),
            'residual_scale': residual_scale,
            'smoothed_history': df['smoothed'].tolist()
        }

    def train(self, mrr_series: pd.DataFrame,
              yearly_seasonality: bool = True,
              weekly_seasonality: bool = False) -> None:
        """Train the forecast model, preferring Prophet and falling back safely if needed."""
        logger.info("=" * 60)
        logger.info("TRAINING FORECASTING MODEL")
        logger.info("=" * 60)

        train_df = self.prepare_data(mrr_series)
        self.actual_df = train_df[['ds', 'y']].copy()
        self.train_df = self._build_training_frame(train_df)

        allow_yearly = yearly_seasonality and len(self.train_df) >= 18

        try:
            if Prophet is None:
                raise RuntimeError("Prophet is not installed in the environment")

            self.model = Prophet(
                growth='linear',
                seasonality_mode='additive',
                changepoint_prior_scale=0.03,  # Moderate smoothing
                seasonality_prior_scale=1.0,
                interval_width=0.80,
                n_changepoints=3,
                yearly_seasonality=False,  # Disable yearly seasonality to avoid over-seasonalizing
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=False
            )

            if len(self.train_df) >= 12:
                self.model.add_seasonality(name='monthly', period=30.5, fourier_order=2)  # Reduced order

            logger.info("Fitting Prophet model...")
            self.model.fit(self.train_df)
            self.model_type = 'prophet'
            logger.info("✓ Model training complete with Prophet")

        except Exception as exc:
            logger.warning(
                "Prophet backend unavailable (%s). Using damped trend + seasonality fallback.",
                exc
            )
            self.fallback_params = self._fit_fallback_model(train_df)
            self.model = self.fallback_params
            self.model_type = 'fallback'
            logger.info("✓ Fallback forecasting model ready")

    def _predict_with_fallback(self, periods: int) -> pd.DataFrame:
        """Generate a realistic forecast without Prophet by using damped growth and monthly seasonality."""
        df = self.actual_df.copy()
        params = self.fallback_params.copy()

        smoothed_history = pd.Series(
            params.get('smoothed_history', df['y'].ewm(span=min(5, max(2, len(df))), adjust=False).mean().tolist())
        )
        residual_scale = float(params.get('residual_scale', 0.08))
        monthly_growth = float(params.get('monthly_growth', 0.03))
        seasonal_factors = params.get('seasonal_factors', {})

        rows = []
        for hist_date, actual_value, smoothed_value in zip(df['ds'], df['y'], smoothed_history):
            band = max(smoothed_value * residual_scale, smoothed_value * 0.04)
            rows.append({
                'ds': hist_date,
                'yhat': float(smoothed_value),
                'yhat_lower': max(float(smoothed_value - band), 0.0),
                'yhat_upper': float(smoothed_value + band)
            })

        recent_run_rate = float(df['y'].tail(min(3, len(df))).mean())
        base_level = max(
            float(smoothed_history.iloc[-1]),
            recent_run_rate * 0.98,
            float(df['y'].iloc[-1]) * 0.99
        )
        last_date = df['ds'].iloc[-1]
        current_factor = seasonal_factors.get(int(last_date.month), 1.0) or 1.0
        cumulative_growth = 1.0

        for step in range(1, periods + 1):
            future_date = last_date + pd.offsets.MonthBegin(step)
            step_growth = 1 + monthly_growth * (0.88 ** (step - 1))
            cumulative_growth *= max(step_growth, 0.97)

            future_factor = seasonal_factors.get(int(future_date.month), 1.0)
            seasonal_adjustment = future_factor / current_factor
            yhat = max(base_level * cumulative_growth * seasonal_adjustment, 0.0)

            interval_pct = residual_scale * (1 + 0.15 * (step - 1))
            band = max(yhat * interval_pct, yhat * 0.05)
            rows.append({
                'ds': future_date,
                'yhat': float(yhat),
                'yhat_lower': max(float(yhat - band), 0.0),
                'yhat_upper': float(yhat + band)
            })

        return pd.DataFrame(rows)

    def predict(self, periods: int = 3) -> pd.DataFrame:
        """Generate forecast for future monthly periods."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info(f"Generating {periods}-month forecast...")

        if self.model_type == 'prophet':
            future = self.model.make_future_dataframe(periods=periods, freq='MS')
            # Only set cap/floor for logistic growth
            if hasattr(self.model, 'growth') and self.model.growth == 'logistic':
                future['cap'] = self.train_df['cap'].iloc[-1]
                future['floor'] = self.train_df['floor'].iloc[-1]
            forecast = self.model.predict(future)
        else:
            forecast = self._predict_with_fallback(periods)

        self.forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            self.forecast_df[col] = pd.to_numeric(self.forecast_df[col], errors='coerce').clip(lower=0)

        logger.info(f"✓ Forecast generated for {periods} months using {self.model_type}")
        return self.forecast_df

    def get_forecast_summary(self) -> pd.DataFrame:
        """Get clean forecast summary with actuals, predictions, and bounds."""
        if self.forecast_df is None:
            raise ValueError("No forecast available. Call predict() first.")

        summary = self.forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        summary.columns = ['date', 'predicted', 'lower_bound', 'upper_bound']

        if self.actual_df is not None:
            actual = self.actual_df[['ds', 'y']].copy()
            actual.columns = ['date', 'actual']
            summary = summary.merge(actual, on='date', how='left')

        for col in ['predicted', 'lower_bound', 'upper_bound', 'actual']:
            if col in summary.columns:
                summary[col] = summary[col].round(2)

        return summary.sort_values('date').reset_index(drop=True)

    def get_future_forecast(self, months_ahead: int = 3) -> pd.DataFrame:
        """Get only the future forecast rows."""
        summary = self.get_forecast_summary()
        last_actual_date = self.actual_df['ds'].max()
        return summary[summary['date'] > last_actual_date].head(months_ahead).reset_index(drop=True)

    def calculate_metrics(self) -> Dict:
        """Calculate in-sample forecast accuracy metrics."""
        if self.forecast_df is None or self.actual_df is None:
            raise ValueError("Need both forecast and actual data for metrics")

        comparison = self.actual_df.merge(
            self.forecast_df[['ds', 'yhat']],
            on='ds',
            how='inner'
        )

        actual = comparison['y'].values
        predicted = comparison['yhat'].values
        denominator = np.where(actual == 0, 1, actual)

        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / denominator)) * 100

        self.metrics = {
            'MAE': round(float(mae), 2),
            'RMSE': round(float(rmse), 2),
            'MAPE': round(float(mape), 2)
        }

        logger.info(
            f"✓ Forecast Metrics ({self.model_type}): MAE=${mae:,.2f}, RMSE=${rmse:,.2f}, MAPE={mape:.2f}%"
        )
        return self.metrics

    def save_model(self, filepath: str = None) -> str:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        if filepath is None:
            filepath = MODELS_DIR / "forecast_model.pkl"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'actual_df': self.actual_df,
            'train_df': self.train_df,
            'forecast_df': self.forecast_df,
            'fallback_params': self.fallback_params,
            'metrics': self.metrics,
            'seasonality_mode': self.seasonality_mode
        }

        joblib.dump(model_data, filepath)
        logger.info(f"✓ Model saved to {filepath}")
        return str(filepath)

    @classmethod
    def load_model(cls, filepath: str = None) -> 'MRRForecaster':
        """Load a trained model from disk."""
        if filepath is None:
            filepath = MODELS_DIR / "forecast_model.pkl"
        else:
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        forecaster = cls(seasonality_mode=model_data['seasonality_mode'])
        forecaster.model = model_data['model']
        forecaster.model_type = model_data.get('model_type', 'prophet')
        forecaster.actual_df = model_data['actual_df']
        forecaster.train_df = model_data.get('train_df')
        forecaster.forecast_df = model_data['forecast_df']
        forecaster.fallback_params = model_data.get('fallback_params', {})
        forecaster.metrics = model_data.get('metrics', {})

        logger.info(f"✓ Model loaded from {filepath}")
        return forecaster

    def plot_forecast(self, show_components: bool = False):
        """Generate forecast plot, supporting both Prophet and fallback forecasts."""
        if self.forecast_df is None:
            raise ValueError("Forecast needed for plotting")

        import matplotlib.pyplot as plt

        if self.model_type == 'prophet' and hasattr(self.model, 'plot'):
            fig = self.model.plot(self.forecast_df)
            plt.title('MRR Forecast')
            plt.ylabel('MRR ($)')
            plt.xlabel('Date')

            if show_components and hasattr(self.model, 'plot_components'):
                self.model.plot_components(self.forecast_df)

            return fig

        summary = self.get_forecast_summary()
        fig, ax = plt.subplots(figsize=(10, 5))
        if 'actual' in summary.columns:
            ax.plot(summary['date'], summary['actual'], label='Actual MRR', color='#1f77b4', linewidth=2)
        ax.plot(summary['date'], summary['predicted'], label='Forecast', color='#ff7f0e', linewidth=2)
        ax.fill_between(summary['date'], summary['lower_bound'], summary['upper_bound'], color='#ff7f0e', alpha=0.2)
        ax.set_title('MRR Forecast')
        ax.set_ylabel('MRR ($)')
        ax.set_xlabel('Date')
        ax.legend()
        plt.tight_layout()
        return fig


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