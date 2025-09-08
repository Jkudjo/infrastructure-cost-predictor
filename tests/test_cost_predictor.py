"""
Tests for Cost Predictor Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_models.cost_predictor import CostPredictor


class TestCostPredictor:
    """Test cases for CostPredictor"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create realistic cost patterns
        base_cost = 8000
        seasonal_pattern = 1000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        trend = np.arange(len(dates)) * 5
        noise = np.random.normal(0, 200, len(dates))
        
        total_cost = base_cost + seasonal_pattern + trend + noise
        
        data = pd.DataFrame({
            'date': dates,
            'total_cost': total_cost,
            'ec2_cost': total_cost * 0.6,
            's3_cost': total_cost * 0.15,
            'rds_cost': total_cost * 0.2,
            'other_cost': total_cost * 0.05,
            'year': dates.year,
            'month': dates.month,
            'day_of_week': dates.dayofweek,
            'day_of_year': dates.dayofyear,
            'is_weekend': (dates.dayofweek >= 5).astype(int)
        })
        
        return data
    
    @pytest.fixture
    def cost_predictor(self):
        """Create CostPredictor instance"""
        return CostPredictor()
    
    def test_initialization(self, cost_predictor):
        """Test CostPredictor initialization"""
        assert cost_predictor.model_dir is not None
        assert cost_predictor.training_data is None
        assert cost_predictor.feature_columns is None
        assert cost_predictor.target_column == "total_cost"
    
    def test_preprocess_data(self, cost_predictor, sample_data):
        """Test data preprocessing"""
        processed_data = cost_predictor._preprocess_data(sample_data)
        
        # Check that preprocessing adds new features
        assert len(processed_data.columns) > len(sample_data.columns)
        
        # Check that lag features are created
        assert 'total_cost_lag_1' in processed_data.columns
        assert 'total_cost_lag_7' in processed_data.columns
        assert 'total_cost_lag_30' in processed_data.columns
        
        # Check that rolling statistics are created
        assert 'total_cost_rolling_mean_7' in processed_data.columns
        assert 'total_cost_rolling_std_30' in processed_data.columns
        
        # Check that feature columns are set
        assert cost_predictor.feature_columns is not None
        assert len(cost_predictor.feature_columns) > 0
    
    def test_create_lstm_data(self, cost_predictor, sample_data):
        """Test LSTM data creation"""
        processed_data = cost_predictor._preprocess_data(sample_data)
        cost_predictor.training_data = processed_data
        
        X, y = cost_predictor._create_lstm_data(processed_data, sequence_length=30)
        
        # Check shapes
        assert X.shape[1] == 30  # sequence length
        assert X.shape[2] == len(cost_predictor.feature_columns)
        assert len(y) == X.shape[0]
        
        # Check that we have enough data
        assert len(X) > 0
    
    def test_build_lstm_model(self, cost_predictor):
        """Test LSTM model building"""
        input_shape = (30, 10)  # sequence_length, features
        model = cost_predictor._build_lstm_model(input_shape)
        
        # Check model structure
        assert model is not None
        assert len(model.layers) == 6  # 2 LSTM + 2 Dropout + 2 Dense
        
        # Check input shape
        assert model.input_shape == (None, 30, 10)
        
        # Check output shape
        assert model.output_shape == (None, 1)
    
    @pytest.mark.asyncio
    async def test_train_models(self, cost_predictor, sample_data):
        """Test model training"""
        processed_data = cost_predictor._preprocess_data(sample_data)
        cost_predictor.training_data = processed_data
        
        # Train models
        scores = await cost_predictor.train(test_size=0.2)
        
        # Check that models are trained
        assert cost_predictor.rf_model is not None
        assert cost_predictor.scaler is not None
        
        # Check that scores are calculated
        assert 'random_forest' in scores
        assert 'mae' in scores['random_forest']
        assert 'mse' in scores['random_forest']
        assert 'r2' in scores['random_forest']
        
        # Check that scores are reasonable
        assert 0 <= scores['random_forest']['r2'] <= 1
    
    @pytest.mark.asyncio
    async def test_predict(self, cost_predictor, sample_data):
        """Test cost prediction"""
        processed_data = cost_predictor._preprocess_data(sample_data)
        cost_predictor.training_data = processed_data
        
        # Train models first
        await cost_predictor.train(test_size=0.2)
        
        # Make prediction
        prediction = await cost_predictor.predict(days=30)
        
        # Check prediction structure
        assert 'predictions' in prediction
        assert 'total_cost' in prediction
        assert 'daily_average' in prediction
        assert 'confidence' in prediction
        assert 'date_range' in prediction
        
        # Check prediction values
        assert len(prediction['predictions']) == 30
        assert prediction['total_cost'] > 0
        assert prediction['daily_average'] > 0
        assert 0 <= prediction['confidence'] <= 1
        
        # Check date range
        assert 'start' in prediction['date_range']
        assert 'end' in prediction['date_range']
    
    def test_get_confidence_score(self, cost_predictor):
        """Test confidence score calculation"""
        # Test with no models
        assert cost_predictor.get_confidence_score() == 0.0
        
        # Test with mock scores
        cost_predictor.model_scores = {
            'random_forest': {'r2': 0.8},
            'lstm': {'r2': 0.7}
        }
        
        confidence = cost_predictor.get_confidence_score()
        assert 0 <= confidence <= 1
        assert confidence == 0.75  # Average of 0.8 and 0.7
    
    def test_get_model_info(self, cost_predictor, sample_data):
        """Test model info retrieval"""
        # Test with no data
        info = cost_predictor.get_model_info()
        assert info['models_trained'] == []
        assert info['features_count'] == 0
        assert info['training_data_size'] == 0
        
        # Test with data and models
        processed_data = cost_predictor._preprocess_data(sample_data)
        cost_predictor.training_data = processed_data
        cost_predictor.model_scores = {'random_forest': {'r2': 0.8}}
        
        info = cost_predictor.get_model_info()
        assert 'random_forest' in info['models_trained']
        assert info['features_count'] > 0
        assert info['training_data_size'] == len(processed_data)
        assert 'random_forest' in info['scores']
    
    @pytest.mark.asyncio
    async def test_save_and_load_models(self, cost_predictor, sample_data, tmp_path):
        """Test model saving and loading"""
        # Set custom model directory
        cost_predictor.model_dir = tmp_path
        
        # Prepare data and train models
        processed_data = cost_predictor._preprocess_data(sample_data)
        cost_predictor.training_data = processed_data
        await cost_predictor.train(test_size=0.2)
        
        # Save models
        await cost_predictor.save_models()
        
        # Check that files are created
        assert (tmp_path / "random_forest.pkl").exists()
        assert (tmp_path / "scaler.pkl").exists()
        assert (tmp_path / "metadata.pkl").exists()
        
        # Create new predictor and load models
        new_predictor = CostPredictor(model_dir=str(tmp_path))
        await new_predictor.load_models()
        
        # Check that models are loaded
        assert new_predictor.rf_model is not None
        assert new_predictor.scaler is not None
        assert new_predictor.feature_columns is not None
        assert len(new_predictor.model_scores) > 0


class TestDataPreprocessing:
    """Test data preprocessing functions"""
    
    def test_handle_missing_values(self):
        """Test handling of missing values"""
        cost_predictor = CostPredictor()
        
        # Create data with missing values
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'total_cost': [1000, 1100, None, 1200, None, 1300, 1400, 1500, 1600, 1700],
            'ec2_cost': [600, 660, 720, None, 780, 840, 900, 960, 1020, 1080]
        })
        
        processed_data = cost_predictor._preprocess_data(data)
        
        # Check that missing values are handled
        assert processed_data['total_cost'].isna().sum() == 0
        assert processed_data['ec2_cost'].isna().sum() == 0
    
    def test_time_features_creation(self):
        """Test creation of time-based features"""
        cost_predictor = CostPredictor()
        
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'total_cost': range(1000, 1010)
        })
        
        processed_data = cost_predictor._preprocess_data(data)
        
        # Check that time features are created
        assert 'year' in processed_data.columns
        assert 'month' in processed_data.columns
        assert 'day_of_week' in processed_data.columns
        assert 'day_of_year' in processed_data.columns
        assert 'is_weekend' in processed_data.columns
        
        # Check values
        assert processed_data['year'].iloc[0] == 2023
        assert processed_data['month'].iloc[0] == 1
        assert processed_data['is_weekend'].dtype == int


if __name__ == "__main__":
    pytest.main([__file__])
