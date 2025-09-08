"""
Cost Prediction Models - Machine Learning for Infrastructure Cost Forecasting
"""

import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

logger = logging.getLogger(__name__)


class CostPredictor:
    """
    Advanced cost prediction using multiple ML models
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Models
        self.lstm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        
        # Data
        self.training_data = None
        self.feature_columns = None
        self.target_column = "total_cost"
        
        # Model performance
        self.model_scores = {}
        
    async def load_data(self, data_source: str) -> pd.DataFrame:
        """Load and preprocess cost data"""
        logger.info(f"Loading data from {data_source}")
        
        try:
            if data_source.endswith('.csv'):
                data = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                data = pd.read_json(data_source)
            else:
                # Assume it's a database connection string
                data = await self._load_from_database(data_source)
            
            # Preprocess data
            data = self._preprocess_data(data)
            self.training_data = data
            
            logger.info(f"Loaded {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the cost data for ML models"""
        logger.info("Preprocessing data...")
        
        # Convert date columns
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
        
        # Create time-based features
        if 'date' in data.columns:
            data['year'] = data['date'].dt.year
            data['month'] = data['date'].dt.month
            data['day_of_week'] = data['date'].dt.dayofweek
            data['day_of_year'] = data['date'].dt.dayofyear
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Create lag features for time series
        if self.target_column in data.columns:
            for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month lags
                data[f'{self.target_column}_lag_{lag}'] = data[self.target_column].shift(lag)
        
        # Create rolling statistics
        if self.target_column in data.columns:
            for window in [7, 30]:  # 1 week, 1 month windows
                data[f'{self.target_column}_rolling_mean_{window}'] = (
                    data[self.target_column].rolling(window=window).mean()
                )
                data[f'{self.target_column}_rolling_std_{window}'] = (
                    data[self.target_column].rolling(window=window).std()
                )
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Select feature columns
        self.feature_columns = [
            col for col in data.columns 
            if col not in [self.target_column, 'date', 'id']
        ]
        
        logger.info(f"Created {len(self.feature_columns)} features")
        return data
    
    def _create_lstm_data(self, data: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Create LSTM training data"""
        logger.info(f"Creating LSTM data with sequence length {sequence_length}")
        
        # Prepare features and target
        features = data[self.feature_columns].values
        target = data[self.target_column].values
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model for time series prediction"""
        logger.info("Building LSTM model...")
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def train(self, test_size: float = 0.2) -> Dict[str, float]:
        """Train all models"""
        if self.training_data is None:
            raise ValueError("No training data loaded")
        
        logger.info("Starting model training...")
        
        # Split data
        split_idx = int(len(self.training_data) * (1 - test_size))
        train_data = self.training_data[:split_idx]
        test_data = self.training_data[split_idx:]
        
        # Train Random Forest
        await self._train_random_forest(train_data, test_data)
        
        # Train LSTM
        await self._train_lstm(train_data, test_data)
        
        logger.info("Model training completed")
        return self.model_scores
    
    async def _train_random_forest(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Prepare data
        X_train = train_data[self.feature_columns]
        y_train = train_data[self.target_column]
        X_test = test_data[self.feature_columns]
        y_test = test_data[self.target_column]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test_scaled)
        self.model_scores['random_forest'] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Random Forest R²: {self.model_scores['random_forest']['r2']:.3f}")
    
    async def _train_lstm(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        
        sequence_length = 30
        
        # Create LSTM data
        X_train, y_train = self._create_lstm_data(train_data, sequence_length)
        X_test, y_test = self._create_lstm_data(test_data, sequence_length)
        
        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning("Insufficient data for LSTM training")
            return
        
        # Build and train model
        self.lstm_model = self._build_lstm_model((sequence_length, len(self.feature_columns)))
        
        # Train model
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluate
        y_pred = self.lstm_model.predict(X_test, verbose=0).flatten()
        self.model_scores['lstm'] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"LSTM R²: {self.model_scores['lstm']['r2']:.3f}")
    
    async def predict(self, days: int = 30) -> Dict[str, Any]:
        """Predict future costs"""
        if self.training_data is None:
            raise ValueError("No training data available")
        
        logger.info(f"Predicting costs for next {days} days")
        
        # Get latest data for prediction
        latest_data = self.training_data.tail(30).copy()  # Last 30 days
        
        predictions = []
        confidence_scores = []
        
        for day in range(days):
            # Prepare features for prediction
            features = latest_data[self.feature_columns].iloc[-1:].values
            
            # Random Forest prediction
            if self.rf_model:
                features_scaled = self.scaler.transform(features)
                rf_pred = self.rf_model.predict(features_scaled)[0]
                predictions.append(rf_pred)
            else:
                predictions.append(0)
            
            # Update data for next prediction (simple approach)
            # In production, you'd want more sophisticated forecasting
            new_row = latest_data.iloc[-1:].copy()
            new_row[self.target_column] = predictions[-1]
            latest_data = pd.concat([latest_data, new_row], ignore_index=True)
        
        # Calculate confidence based on model performance
        avg_r2 = np.mean([score['r2'] for score in self.model_scores.values()])
        confidence = max(0, min(1, avg_r2))
        
        return {
            'predictions': predictions,
            'total_cost': sum(predictions),
            'daily_average': np.mean(predictions),
            'confidence': confidence,
            'date_range': {
                'start': (datetime.now() + timedelta(days=1)).isoformat(),
                'end': (datetime.now() + timedelta(days=days)).isoformat()
            }
        }
    
    def get_confidence_score(self) -> float:
        """Get overall model confidence score"""
        if not self.model_scores:
            return 0.0
        
        avg_r2 = np.mean([score['r2'] for score in self.model_scores.values()])
        return max(0, min(1, avg_r2))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        return {
            'models_trained': list(self.model_scores.keys()),
            'scores': self.model_scores,
            'features_count': len(self.feature_columns) if self.feature_columns else 0,
            'training_data_size': len(self.training_data) if self.training_data is not None else 0
        }
    
    async def save_models(self):
        """Save trained models"""
        logger.info("Saving models...")
        
        # Save Random Forest
        if self.rf_model:
            joblib.dump(self.rf_model, self.model_dir / "random_forest.pkl")
            joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
        
        # Save LSTM
        if self.lstm_model:
            self.lstm_model.save(self.model_dir / "lstm_model")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'model_scores': self.model_scores,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.model_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info("Models saved successfully")
    
    async def load_models(self):
        """Load previously trained models"""
        logger.info("Loading saved models...")
        
        try:
            # Load metadata
            with open(self.model_dir / "metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.model_scores = metadata['model_scores']
            
            # Load Random Forest
            if (self.model_dir / "random_forest.pkl").exists():
                self.rf_model = joblib.load(self.model_dir / "random_forest.pkl")
                self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            
            # Load LSTM
            if (self.model_dir / "lstm_model").exists():
                self.lstm_model = tf.keras.models.load_model(self.model_dir / "lstm_model")
            
            logger.info("Models loaded successfully")
            
        except FileNotFoundError:
            logger.info("No saved models found")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    async def _load_from_database(self, connection_string: str) -> pd.DataFrame:
        """Load data from database (placeholder implementation)"""
        # This would be implemented based on your database setup
        # For now, return sample data
        logger.warning("Database loading not implemented, using sample data")
        
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'total_cost': np.random.normal(1000, 200, len(dates)),
            'ec2_cost': np.random.normal(600, 150, len(dates)),
            's3_cost': np.random.normal(200, 50, len(dates)),
            'rds_cost': np.random.normal(200, 50, len(dates))
        })
        
        return sample_data
