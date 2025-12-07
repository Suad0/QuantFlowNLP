"""Unit tests for prediction service.

This module tests the PredictionService functionality including model loading,
validation, and inference.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.core.exceptions import PredictionError
from app.models.xlstm_model import LightweightXLSTM
from app.services.model_loader import ModelLoader
from app.services.prediction import PredictionService


class TestPredictionService:
    """Test suite for PredictionService."""
    
    @pytest.fixture
    def prediction_service(self):
        """Create a prediction service instance."""
        return PredictionService()
    
    @pytest.fixture
    def sample_feature_sequence(self):
        """Create a sample feature sequence for testing."""
        # Shape: [60, 6] - 60 timesteps, 6 features
        return np.random.randn(60, 6).astype(np.float32)
    
    def test_service_initialization(self, prediction_service):
        """Test that service initializes correctly."""
        assert prediction_service is not None
        assert not prediction_service.is_model_loaded()
        assert prediction_service.model is None
        assert prediction_service.model_version is None
    
    @pytest.mark.asyncio
    async def test_predict_without_model_fails(self, prediction_service, sample_feature_sequence):
        """Test that prediction fails when no model is loaded."""
        with pytest.raises(PredictionError, match="No model loaded"):
            await prediction_service.predict(sample_feature_sequence)
    
    def test_validate_input_correct_shape(self, prediction_service):
        """Test input validation with correct shape."""
        valid_input = np.random.randn(60, 6).astype(np.float32)
        
        # Should not raise
        prediction_service._validate_input(valid_input)
    
    def test_validate_input_wrong_dimensions(self, prediction_service):
        """Test input validation with wrong dimensions."""
        # 1D array
        invalid_input = np.random.randn(60).astype(np.float32)
        
        with pytest.raises(PredictionError, match="Expected 2D array"):
            prediction_service._validate_input(invalid_input)
    
    def test_validate_input_wrong_features(self, prediction_service):
        """Test input validation with wrong number of features."""
        # Wrong number of features (5 instead of 6)
        invalid_input = np.random.randn(60, 5).astype(np.float32)
        
        with pytest.raises(PredictionError, match="Expected 6 features"):
            prediction_service._validate_input(invalid_input)
    
    def test_validate_input_nan_values(self, prediction_service):
        """Test input validation with NaN values."""
        invalid_input = np.random.randn(60, 6).astype(np.float32)
        invalid_input[0, 0] = np.nan
        
        with pytest.raises(PredictionError, match="NaN values"):
            prediction_service._validate_input(invalid_input)
    
    def test_validate_input_inf_values(self, prediction_service):
        """Test input validation with infinite values."""
        invalid_input = np.random.randn(60, 6).astype(np.float32)
        invalid_input[0, 0] = np.inf
        
        with pytest.raises(PredictionError, match="infinite values"):
            prediction_service._validate_input(invalid_input)


class TestModelLoader:
    """Test suite for ModelLoader."""
    
    @pytest.fixture
    def model_loader(self):
        """Create a model loader instance."""
        return ModelLoader()
    
    def test_model_loader_initialization(self, model_loader):
        """Test that model loader initializes correctly."""
        assert model_loader is not None
        assert model_loader.expected_num_features == 6
        assert len(model_loader.expected_features) == 6
    
    def test_get_model_requirements(self, model_loader):
        """Test getting model requirements."""
        requirements = model_loader.get_model_requirements()
        
        assert requirements["num_features"] == 6
        assert requirements["expected_sequence_length"] == 60
        assert requirements["output_range"] == "[-1, 1]"
    
    def test_validate_model_with_correct_config(self, model_loader):
        """Test model validation with correct configuration."""
        model = LightweightXLSTM(
            input_features=6,
            hidden_dim=32,
            num_layers=2,
            attention_heads=4,
        )
        
        config = {
            "input_features": ["open", "high", "low", "close", "volume", "news_score"],
            "sequence_length": 60,
            "hidden_dim": 32,
            "num_layers": 2,
            "attention_heads": 4,
        }
        
        # Should not raise
        model_loader.validate_model(model, config)
    
    def test_validate_model_wrong_features(self, model_loader):
        """Test model validation with wrong number of features."""
        model = LightweightXLSTM(
            input_features=5,  # Wrong number
            hidden_dim=32,
            num_layers=2,
            attention_heads=4,
        )
        
        with pytest.raises(PredictionError, match="expects 5 features"):
            model_loader.validate_model(model)


class TestLightweightXLSTM:
    """Test suite for LightweightXLSTM model."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance."""
        return LightweightXLSTM(
            input_features=6,
            hidden_dim=32,
            num_layers=2,
            attention_heads=4,
        )
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None
        assert model.input_features == 6
        assert model.hidden_dim == 32
        assert model.num_layers == 2
        assert model.attention_heads == 4
    
    def test_forward_pass(self, model):
        """Test forward pass through the model."""
        # Create dummy input: [batch=2, seq_len=60, features=6]
        x = torch.randn(2, 60, 6)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (2, 1)
        
        # Check output range [-1, 1]
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)
    
    def test_predict_method(self, model):
        """Test predict method."""
        x = torch.randn(1, 60, 6)
        
        # Predict
        output = model.predict(x)
        
        # Check output
        assert output.shape == (1, 1)
        assert -1.0 <= output.item() <= 1.0
    
    def test_get_model_info(self, model):
        """Test getting model information."""
        info = model.get_model_info()
        
        assert info["model_type"] == "LightweightXLSTM"
        assert info["input_features"] == 6
        assert info["hidden_dim"] == 32
        assert info["num_layers"] == 2
        assert info["attention_heads"] == 4
        assert "total_parameters" in info
        assert "trainable_parameters" in info
    
    def test_model_output_consistency(self, model):
        """Test that model produces consistent outputs in eval mode."""
        x = torch.randn(1, 60, 6)
        
        model.eval()
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        # Outputs should be identical in eval mode with same input
        assert torch.allclose(output1, output2)
