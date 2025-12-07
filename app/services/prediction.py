"""Prediction service for model inference.

This module provides the PredictionService which manages model loading,
validation, and inference for generating trade scores.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.core.config import settings
from app.core.exceptions import PredictionError
from app.models.domain import PredictionResult
from app.models.xlstm_model import LightweightXLSTM
from app.services.model_loader import ModelLoader
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Service for managing model inference and predictions.
    
    This service handles:
    - Loading and unloading PyTorch models
    - Validating input data
    - Generating predictions
    - Managing model state
    """
    
    def __init__(self):
        """Initialize the prediction service."""
        self.model: Optional[LightweightXLSTM] = None
        self.model_loader = ModelLoader()
        self.model_version: Optional[str] = None
        self.model_config: Optional[dict] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"PredictionService initialized with device: {self.device}")
    
    async def predict(
        self,
        feature_sequence: np.ndarray,
        symbol: Optional[str] = None,
    ) -> PredictionResult:
        """Generate prediction from feature sequence.
        
        Args:
            feature_sequence: Feature matrix [seq_len, num_features]
            symbol: Optional trading symbol for metadata
        
        Returns:
            PredictionResult with trade score and metadata
        
        Raises:
            PredictionError: If prediction fails or model not loaded
        """
        try:
            # Check if model is loaded
            if not self.is_model_loaded():
                raise PredictionError(
                    "No model loaded. Please load a model first using load_model()"
                )
            
            logger.debug(
                f"Generating prediction for sequence shape: {feature_sequence.shape}"
            )
            
            # Validate input
            self._validate_input(feature_sequence)
            
            # Convert to tensor and add batch dimension
            # Input shape: [seq_len, features] -> [1, seq_len, features]
            input_tensor = torch.from_numpy(feature_sequence).float()
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)
            
            # Generate prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Extract trade score
            trade_score = output.item()
            
            # Calculate confidence (simple heuristic based on magnitude)
            # Higher magnitude = higher confidence
            confidence = abs(trade_score)
            
            logger.info(
                f"Prediction generated: trade_score={trade_score:.4f}, "
                f"confidence={confidence:.4f}"
            )
            
            # Create result
            result = PredictionResult(
                trade_score=trade_score,
                confidence=confidence,
                model_version=self.model_version or "unknown",
                predicted_at=datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow(),
                metadata={
                    "symbol": symbol,
                    "input_shape": list(feature_sequence.shape),
                    "device": self.device,
                    "model_config": self.model_config,
                },
            )
            
            return result
            
        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {e}") from e
    
    async def load_model(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Load a PyTorch model from disk.
        
        If no paths are provided, loads the default model from
        model_downloaded/xlstm_forecaster.pth with config.json.
        
        Args:
            model_path: Path to model weights file (.pth)
            config_path: Path to model configuration file (.json)
        
        Raises:
            PredictionError: If model loading fails
        """
        try:
            # Use default paths if not provided
            if model_path is None:
                model_path = "model_downloaded/xlstm_forecaster.pth"
            
            if config_path is None:
                # Try to find config in same directory as model
                model_dir = Path(model_path).parent
                config_path = str(model_dir / "config.json")
            
            logger.info(f"Loading model from {model_path}")
            
            # Load configuration first
            if Path(config_path).exists():
                with open(config_path, "r") as f:
                    self.model_config = json.load(f)
                logger.info(f"Loaded model config: {self.model_config}")
            else:
                logger.warning(f"Config file not found: {config_path}")
                self.model_config = None
            
            # Load model using ModelLoader
            self.model = self.model_loader.load_xlstm_model(
                model_path=model_path,
                config_path=config_path if Path(config_path).exists() else None,
                device=self.device,
            )
            
            # Set model version from path
            self.model_version = Path(model_path).stem
            
            logger.info(
                f"Model loaded successfully: {self.model_version}, "
                f"device: {self.device}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.model_version = None
            self.model_config = None
            raise PredictionError(f"Failed to load model: {e}") from e
    
    async def unload_model(self) -> None:
        """Unload the current model and free memory.
        
        This method clears the model from memory and resets the service state.
        """
        try:
            if self.model is not None:
                logger.info(f"Unloading model: {self.model_version}")
                
                # Clear model
                self.model = None
                self.model_version = None
                self.model_config = None
                
                # Clear CUDA cache if using GPU
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info("Model unloaded successfully")
            else:
                logger.debug("No model to unload")
                
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            raise PredictionError(f"Failed to unload model: {e}") from e
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        
        Raises:
            PredictionError: If no model is loaded
        """
        if not self.is_model_loaded():
            raise PredictionError("No model loaded")
        
        info = {
            "model_version": self.model_version,
            "device": self.device,
            "is_loaded": True,
            "config": self.model_config,
        }
        
        if self.model:
            info.update(self.model.get_model_info())
        
        return info
    
    def _validate_input(self, feature_sequence: np.ndarray) -> None:
        """Validate input feature sequence.
        
        Checks:
        - Input is 2D array
        - Number of features matches model expectations (6)
        - Sequence length matches config (60)
        - No NaN or infinite values
        
        Args:
            feature_sequence: Feature matrix to validate
        
        Raises:
            PredictionError: If validation fails
        """
        try:
            # Check dimensions
            if feature_sequence.ndim != 2:
                raise PredictionError(
                    f"Expected 2D array, got {feature_sequence.ndim}D"
                )
            
            seq_len, num_features = feature_sequence.shape
            
            # Check number of features
            expected_features = 6
            if num_features != expected_features:
                raise PredictionError(
                    f"Expected {expected_features} features, got {num_features}"
                )
            
            # Check sequence length
            if self.model_config:
                expected_seq_len = self.model_config.get("sequence_length", 60)
                if seq_len != expected_seq_len:
                    raise PredictionError(
                        f"Expected sequence length {expected_seq_len}, got {seq_len}"
                    )
            
            # Check for NaN or infinite values
            if np.isnan(feature_sequence).any():
                raise PredictionError("Input contains NaN values")
            
            if np.isinf(feature_sequence).any():
                raise PredictionError("Input contains infinite values")
            
            logger.debug(
                f"Input validation passed: shape={feature_sequence.shape}"
            )
            
        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Input validation failed: {e}") from e
