"""Feature engineering API router.

This module provides REST endpoints for building feature sequences
for model inference.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import FeatureServiceDep
from app.core.exceptions import FeatureEngineeringError
from app.models.api.features import BuildSequenceRequest, BuildSequenceResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/build-sequence",
    response_model=BuildSequenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Build feature sequence for model inference",
    description=(
        "Build a properly formatted feature sequence for xLSTM model inference. "
        "Fetches OHLCV data, aligns news scores, normalizes features, and returns "
        "a matrix ready for prediction."
    ),
)
async def build_sequence(
    request: BuildSequenceRequest,
    feature_service: FeatureServiceDep,
) -> BuildSequenceResponse:
    """Build feature sequence for model inference.
    
    This endpoint orchestrates the complete feature engineering pipeline:
    1. Fetch OHLCV data for the specified symbol
    2. Resample to target frequency
    3. Align news scores with timestamps
    4. Normalize features using stored parameters
    5. Return properly shaped matrix
    
    Args:
        request: Feature sequence request
        feature_service: Feature engineering service (injected)
    
    Returns:
        Feature sequence ready for model inference
    
    Raises:
        HTTPException: If feature building fails
    """
    logger.info(
        f"Building feature sequence for {request.symbol}",
        extra={
            "symbol": request.symbol,
            "sequence_length": request.sequence_length,
            "end_time": request.end_time,
        },
    )
    
    try:
        # Build feature sequence
        feature_sequence = await feature_service.build_sequence(
            symbol=request.symbol,
            sequence_length=request.sequence_length,
            end_time=request.end_time,
        )
        
        logger.info(
            f"Feature sequence built successfully for {request.symbol}",
            extra={
                "shape": feature_sequence.sequence.shape,
                "feature_count": len(feature_sequence.feature_names),
            },
        )
        
        # Convert numpy array to list for JSON serialization
        sequence_list = feature_sequence.sequence.tolist()
        
        # Convert to API response model
        return BuildSequenceResponse(
            symbol=feature_sequence.symbol,
            sequence=sequence_list,
            feature_names=feature_sequence.feature_names,
            timestamps=feature_sequence.timestamps,
            shape=(
                feature_sequence.sequence.shape[0],
                feature_sequence.sequence.shape[1],
            ),
            metadata=feature_sequence.metadata,
        )
        
    except FeatureEngineeringError as e:
        logger.error(f"Feature engineering failed for {request.symbol}: {e}")
        
        # Check for specific error types
        if "no data" in str(e).lower() or "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for symbol {request.symbol}: {str(e)}",
            ) from e
        elif "scaler" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Scaler parameters not found for {request.symbol}. "
                       f"Please ensure scaler parameters are loaded: {str(e)}",
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feature engineering failed: {str(e)}",
            ) from e
            
    except ValueError as e:
        logger.error(f"Invalid input for feature building: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(
            f"Unexpected error building features for {request.symbol}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e
