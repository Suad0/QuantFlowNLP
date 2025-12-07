"""Prediction API router.

This module provides REST endpoints for model loading and prediction generation.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import PredictionServiceDep
from app.core.exceptions import PredictionError
from app.models.api.prediction import (
    LoadModelRequest,
    LoadModelResponse,
    ModelStatusResponse,
    PredictRequest,
    PredictResponse,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/predict-final-score",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate prediction from feature sequence",
    description=(
        "Generate a trade score prediction from a feature sequence using the loaded xLSTM model. "
        "The feature sequence must be properly formatted with shape [seq_len, 6] where features "
        "are [open, high, low, close, volume, news_score]."
    ),
)
async def predict_final_score(
    request: PredictRequest,
    prediction_service: PredictionServiceDep,
) -> PredictResponse:
    """Generate prediction from feature sequence.
    
    Takes a feature sequence matrix and generates a trade score prediction
    using the loaded xLSTM model. The trade score is in range [-1, 1] where:
    - Positive values indicate bullish signals
    - Negative values indicate bearish signals
    - Magnitude indicates strength of signal
    
    Args:
        request: Prediction request with feature sequence
        prediction_service: Prediction service (injected)
    
    Returns:
        Prediction result with trade score and confidence
    
    Raises:
        HTTPException: If prediction fails or model not loaded
    """
    logger.info(
        "Received prediction request",
        extra={
            "symbol": request.symbol,
            "sequence_shape": (
                len(request.feature_sequence),
                len(request.feature_sequence[0]) if request.feature_sequence else 0,
            ),
        },
    )
    
    try:
        # Convert list to numpy array
        import numpy as np
        feature_array = np.array(request.feature_sequence, dtype=np.float32)
        
        # Generate prediction
        result = await prediction_service.predict(
            feature_sequence=feature_array,
            symbol=request.symbol,
        )
        
        logger.info(
            "Prediction generated successfully",
            extra={
                "trade_score": result.trade_score,
                "confidence": result.confidence,
                "model_version": result.model_version,
            },
        )
        
        # Convert to API response model
        return PredictResponse(
            trade_score=result.trade_score,
            confidence=result.confidence,
            model_version=result.model_version,
            predicted_at=result.predicted_at,
            metadata=result.metadata,
        )
        
    except PredictionError as e:
        logger.error(f"Prediction failed: {e}")
        
        # Check for specific error types
        if "not loaded" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded. Please load a model first using POST /model/load",
            ) from e
        elif "invalid" in str(e).lower() or "shape" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input: {str(e)}",
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            ) from e
            
    except ValueError as e:
        logger.error(f"Invalid input for prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.post(
    "/model/load",
    response_model=LoadModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Load prediction model",
    description=(
        "Load a trained xLSTM model for inference. If no paths are provided, "
        "loads the default model from the configured model directory."
    ),
)
async def load_model(
    request: LoadModelRequest,
    prediction_service: PredictionServiceDep,
) -> LoadModelResponse:
    """Load a prediction model.
    
    Loads a trained PyTorch model and its configuration for inference.
    The model must be compatible with the LightweightXLSTM architecture.
    
    Args:
        request: Model loading request with optional paths
        prediction_service: Prediction service (injected)
    
    Returns:
        Model loading status and information
    
    Raises:
        HTTPException: If model loading fails
    """
    logger.info(
        "Loading model",
        extra={
            "model_path": request.model_path,
            "config_path": request.config_path,
        },
    )
    
    try:
        # Load model
        await prediction_service.load_model(
            model_path=request.model_path,
            config_path=request.config_path,
        )
        
        # Get model info
        model_info = {
            "model_version": prediction_service.model_version,
            "device": prediction_service.device,
            "config": prediction_service.model_config,
        }
        
        logger.info(
            "Model loaded successfully",
            extra={"model_version": prediction_service.model_version},
        )
        
        return LoadModelResponse(
            success=True,
            message="Model loaded successfully",
            model_info=model_info,
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found: {str(e)}",
        ) from e
        
    except PredictionError as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.get(
    "/model/status",
    response_model=ModelStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get model status",
    description="Check if a model is currently loaded and get model information.",
)
async def get_model_status(
    prediction_service: PredictionServiceDep,
) -> ModelStatusResponse:
    """Get current model status.
    
    Returns information about whether a model is loaded and its details.
    
    Args:
        prediction_service: Prediction service (injected)
    
    Returns:
        Model status information
    """
    logger.debug("Checking model status")
    
    is_loaded = prediction_service.is_model_loaded()
    
    if is_loaded:
        model_info = {
            "config": prediction_service.model_config,
        }
        
        return ModelStatusResponse(
            is_loaded=True,
            model_version=prediction_service.model_version,
            device=prediction_service.device,
            model_info=model_info,
        )
    else:
        return ModelStatusResponse(
            is_loaded=False,
            model_version=None,
            device=None,
            model_info=None,
        )
