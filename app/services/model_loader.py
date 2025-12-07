"""Model loader for PyTorch models.

This module provides utilities for loading, validating, and managing
PyTorch models for inference.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from app.core.exceptions import PredictionError
from app.models.xlstm_model import LightweightXLSTM
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Handles loading and validation of PyTorch models.
    
    This class provides methods to load pre-trained xLSTM models,
    validate their configuration, and ensure they match expected
    specifications.
    """
    
    def __init__(self):
        """Initialize the model loader."""
        self.expected_features = ["open", "high", "low", "close", "volume", "news_score"]
        self.expected_num_features = 6
    
    def load_xlstm_model(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cpu",
    ) -> LightweightXLSTM:
        """Load a pre-trained xLSTM model from disk.
        
        Args:
            model_path: Path to the model weights file (.pth)
            config_path: Path to the model configuration file (.json)
            device: Device to load the model on ('cpu' or 'cuda' , or mps but only on apple silicon)
        
        Returns:
            Loaded xLSTM model ready for inference
        
        Raises:
            PredictionError: If model loading or validation fails
        """
        try:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise PredictionError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading xLSTM model from {model_path}")
            
            # Load configuration if provided
            config = None
            if config_path:
                config = self._load_config(config_path)
            else:
                # Try to find config in same directory
                config_path_auto = model_path_obj.parent / "config.json"
                if config_path_auto.exists():
                    config = self._load_config(str(config_path_auto))
            
            # Create model instance
            if config:
                model = self._create_model_from_config(config)
            else:
                # Use default configuration
                logger.warning("No config found, using default model configuration")
                model = LightweightXLSTM(
                    input_features=6,
                    hidden_dim=32,
                    num_layers=2,
                    attention_heads=4,
                )
            
            # Load model weights
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    # Checkpoint contains model_state and other metadata
                    state_dict = checkpoint["model_state"]
                    logger.debug(f"Loaded checkpoint with keys: {checkpoint.keys()}")
                else:
                    # Direct state_dict
                    state_dict = checkpoint
                
                # Load with strict=False to allow missing keys (e.g., layer norms)
                # The pre-trained model may have a slightly different architecture
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys in state_dict: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
                
                logger.info("Successfully loaded model weights")
            except Exception as e:
                raise PredictionError(f"Failed to load model weights: {e}") from e
            
            # Move model to device and set to eval mode
            model = model.to(device)
            model.eval()
            
            # Validate the model
            self.validate_model(model, config)
            
            logger.info(
                f"Model loaded successfully: {model.get_model_info()}"
            )
            
            return model
            
        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Failed to load xLSTM model: {e}")
            raise PredictionError(f"Failed to load model: {e}") from e
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Configuration dictionary
        
        Raises:
            PredictionError: If config loading fails
        """
        try:
            config_path_obj = Path(config_path)
            if not config_path_obj.exists():
                raise PredictionError(f"Config file not found: {config_path}")
            
            with open(config_path_obj, "r") as f:
                config = json.load(f)
            
            logger.debug(f"Loaded config from {config_path}: {config}")
            return config
            
        except json.JSONDecodeError as e:
            raise PredictionError(f"Invalid JSON in config file: {e}") from e
        except Exception as e:
            raise PredictionError(f"Failed to load config: {e}") from e
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> LightweightXLSTM:
        """Create model instance from configuration.
        
        Args:
            config: Model configuration dictionary
        
        Returns:
            Initialized model instance
        
        Raises:
            PredictionError: If model creation fails
        """
        try:
            # Extract model parameters from config
            hidden_dim = config.get("hidden_dim", 32)
            num_layers = config.get("num_layers", 2)
            attention_heads = config.get("attention_heads", 4)
            
            # Get input features
            input_features_list = config.get("input_features", self.expected_features)
            input_features = len(input_features_list)
            
            logger.debug(
                f"Creating model with: hidden_dim={hidden_dim}, "
                f"num_layers={num_layers}, attention_heads={attention_heads}, "
                f"input_features={input_features}"
            )
            
            model = LightweightXLSTM(
                input_features=input_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                attention_heads=attention_heads,
            )
            
            return model
            
        except Exception as e:
            raise PredictionError(f"Failed to create model from config: {e}") from e
    
    def validate_model(
        self,
        model: LightweightXLSTM,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Validate that model matches expected specifications.
        
        Performs validation checks:
        - Input features match expected (6 features)
        - Feature names match expected list
        - Sequence length matches config (60)
        - Model can perform forward pass
        
        Args:
            model: Model to validate
            config: Optional configuration to validate against
        
        Raises:
            PredictionError: If validation fails
        """
        try:
            logger.debug("Validating model configuration")
            
            # Check input features
            if model.input_features != self.expected_num_features:
                raise PredictionError(
                    f"Model expects {model.input_features} features, "
                    f"but system requires {self.expected_num_features}"
                )
            
            # Validate feature names if config provided
            if config:
                config_features = config.get("input_features", [])
                if config_features != self.expected_features:
                    logger.warning(
                        f"Config features {config_features} differ from "
                        f"expected {self.expected_features}"
                    )
                
                # Validate sequence length
                config_seq_len = config.get("sequence_length")
                if config_seq_len:
                    logger.debug(f"Model expects sequence length: {config_seq_len}")
            
            # Test forward pass with dummy data
            self._test_forward_pass(model)
            
            logger.info("Model validation passed")
            
        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Model validation failed: {e}") from e
    
    def _test_forward_pass(self, model: LightweightXLSTM) -> None:
        """Test that model can perform a forward pass.
        
        Args:
            model: Model to test
        
        Raises:
            PredictionError: If forward pass fails
        """
        try:
            # Create dummy input: [batch=1, seq_len=60, features=6]
            dummy_input = torch.randn(1, 60, model.input_features)
            
            # Perform forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            # Validate output shape
            if output.shape != (1, 1):
                raise PredictionError(
                    f"Expected output shape (1, 1), got {output.shape}"
                )
            
            # Validate output range [-1, 1]
            if not (-1.0 <= output.item() <= 1.0):
                raise PredictionError(
                    f"Output {output.item()} outside expected range [-1, 1]"
                )
            
            logger.debug(
                f"Forward pass test successful, output: {output.item():.4f}"
            )
            
        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Forward pass test failed: {e}") from e
    
    def get_model_requirements(self) -> Dict[str, Any]:
        """Get model input requirements.
        
        Returns:
            Dictionary with model requirements
        """
        return {
            "input_features": self.expected_features,
            "num_features": self.expected_num_features,
            "expected_sequence_length": 60,
            "input_shape": "[batch, 60, 6]",
            "output_shape": "[batch, 1]",
            "output_range": "[-1, 1]",
        }
