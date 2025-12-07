"""Integration tests for prediction service with pre-trained model.

This module tests loading and using the actual pre-trained model from
model_downloaded/ directory.
"""

from pathlib import Path

import numpy as np
import pytest

from app.services.prediction import PredictionService


class TestPredictionServiceIntegration:
    """Integration tests for PredictionService with real model."""
    
    @pytest.fixture
    def prediction_service(self):
        """Create a prediction service instance."""
        return PredictionService()
    
    @pytest.fixture
    def model_path(self):
        """Get path to pre-trained model."""
        return "model_downloaded/xlstm_forecaster.pth"
    
    @pytest.fixture
    def config_path(self):
        """Get path to model config."""
        return "model_downloaded/config.json"
    
    @pytest.fixture
    def sample_feature_sequence(self):
        """Create a sample feature sequence matching model expectations."""
        # Shape: [60, 6] - 60 timesteps, 6 features
        # Use realistic values
        np.random.seed(42)
        sequence = np.random.randn(60, 6).astype(np.float32)
        
        # Normalize to reasonable ranges
        sequence[:, 0:4] = sequence[:, 0:4] * 10 + 100  # OHLC around 100
        sequence[:, 4] = np.abs(sequence[:, 4]) * 1000000  # Volume
        sequence[:, 5] = np.clip(sequence[:, 5] * 0.3, -1, 1)  # News score [-1, 1]
        
        return sequence
    
    @pytest.mark.asyncio
    async def test_load_pretrained_model(
        self,
        prediction_service,
        model_path,
        config_path,
    ):
        """Test loading the pre-trained model."""
        # Check if model file exists
        if not Path(model_path).exists():
            pytest.skip(f"Pre-trained model not found at {model_path}")
        
        # Load model
        await prediction_service.load_model(model_path, config_path)
        
        # Verify model is loaded
        assert prediction_service.is_model_loaded()
        assert prediction_service.model is not None
        assert prediction_service.model_version == "xlstm_forecaster"
    
    @pytest.mark.asyncio
    async def test_predict_with_pretrained_model(
        self,
        prediction_service,
        model_path,
        config_path,
        sample_feature_sequence,
    ):
        """Test making predictions with the pre-trained model."""
        # Check if model file exists
        if not Path(model_path).exists():
            pytest.skip(f"Pre-trained model not found at {model_path}")
        
        # Load model
        await prediction_service.load_model(model_path, config_path)
        
        # Make prediction
        result = await prediction_service.predict(
            sample_feature_sequence,
            symbol="TEST",
        )
        
        # Verify result
        assert result is not None
        assert -1.0 <= result.trade_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_version == "xlstm_forecaster"
        assert result.metadata["symbol"] == "TEST"
        assert result.metadata["input_shape"] == [60, 6]
    
    @pytest.mark.asyncio
    async def test_model_info(
        self,
        prediction_service,
        model_path,
        config_path,
    ):
        """Test getting model information."""
        # Check if model file exists
        if not Path(model_path).exists():
            pytest.skip(f"Pre-trained model not found at {model_path}")
        
        # Load model
        await prediction_service.load_model(model_path, config_path)
        
        # Get model info
        info = prediction_service.get_model_info()
        
        # Verify info
        assert info["is_loaded"] is True
        assert info["model_version"] == "xlstm_forecaster"
        assert info["model_type"] == "LightweightXLSTM"
        assert info["input_features"] == 6
        assert info["hidden_dim"] == 32
        assert info["num_layers"] == 2
        assert info["attention_heads"] == 4
    
    @pytest.mark.asyncio
    async def test_unload_model(
        self,
        prediction_service,
        model_path,
        config_path,
    ):
        """Test unloading the model."""
        # Check if model file exists
        if not Path(model_path).exists():
            pytest.skip(f"Pre-trained model not found at {model_path}")
        
        # Load model
        await prediction_service.load_model(model_path, config_path)
        assert prediction_service.is_model_loaded()
        
        # Unload model
        await prediction_service.unload_model()
        
        # Verify model is unloaded
        assert not prediction_service.is_model_loaded()
        assert prediction_service.model is None
        assert prediction_service.model_version is None
    
    @pytest.mark.asyncio
    async def test_multiple_predictions(
        self,
        prediction_service,
        model_path,
        config_path,
        sample_feature_sequence,
    ):
        """Test making multiple predictions with the same model."""
        # Check if model file exists
        if not Path(model_path).exists():
            pytest.skip(f"Pre-trained model not found at {model_path}")
        
        # Load model
        await prediction_service.load_model(model_path, config_path)
        
        # Make multiple predictions
        results = []
        for i in range(3):
            result = await prediction_service.predict(
                sample_feature_sequence,
                symbol=f"TEST{i}",
            )
            results.append(result)
        
        # Verify all results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert -1.0 <= result.trade_score <= 1.0
            assert result.metadata["symbol"] == f"TEST{i}"
        
        # With same input, predictions should be identical (deterministic)
        assert results[0].trade_score == results[1].trade_score
        assert results[1].trade_score == results[2].trade_score
