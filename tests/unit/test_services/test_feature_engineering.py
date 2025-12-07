"""Unit tests for FeatureEngineeringService.

This module tests the feature engineering service functionality including
OHLCV data fetching, resampling, news score alignment, and normalization.
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import FeatureEngineeringError
from app.models.domain import OHLCVData, ScalerParams
from app.services.feature_engineering import FeatureEngineeringService


@pytest.fixture
def mock_ohlcv_repo():
    """Create mock OHLCV repository."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def mock_analysis_repo():
    """Create mock analysis repository."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def mock_scaler_repo():
    """Create mock scaler repository."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def feature_service(mock_ohlcv_repo, mock_analysis_repo, mock_scaler_repo):
    """Create feature engineering service with mocked dependencies."""
    return FeatureEngineeringService(
        ohlcv_repository=mock_ohlcv_repo,
        analysis_repository=mock_analysis_repo,
        scaler_repository=mock_scaler_repo,
    )


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    data = []
    
    for i in range(200):
        data.append(
            OHLCVData(
                id=str(uuid.uuid4()),
                symbol="AAPL",
                timestamp=base_time + timedelta(hours=i),
                open=100.0 + i * 0.1,
                high=101.0 + i * 0.1,
                low=99.0 + i * 0.1,
                close=100.5 + i * 0.1,
                volume=1000000.0 + i * 1000,
            )
        )
    
    return data


@pytest.mark.asyncio
async def test_get_ohlcv_data(feature_service, mock_ohlcv_repo, sample_ohlcv_data):
    """Test OHLCV data fetching."""
    # Arrange
    symbol = "AAPL"
    sequence_length = 168
    end_time = datetime(2024, 1, 10, 0, 0, 0)
    mock_ohlcv_repo.get_range.return_value = sample_ohlcv_data
    
    # Act
    df = await feature_service.get_ohlcv_data(symbol, sequence_length, end_time)
    
    # Assert
    assert not df.empty
    assert len(df) == len(sample_ohlcv_data)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    mock_ohlcv_repo.get_range.assert_called_once()


@pytest.mark.asyncio
async def test_get_ohlcv_data_empty(feature_service, mock_ohlcv_repo):
    """Test OHLCV data fetching with no data."""
    # Arrange
    symbol = "AAPL"
    sequence_length = 168
    end_time = datetime(2024, 1, 10, 0, 0, 0)
    mock_ohlcv_repo.get_range.return_value = []
    
    # Act
    df = await feature_service.get_ohlcv_data(symbol, sequence_length, end_time)
    
    # Assert
    assert df.empty


@pytest.mark.asyncio
async def test_resample_ohlcv(feature_service):
    """Test OHLCV resampling."""
    # Arrange
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(24)]
    
    df = pd.DataFrame({
        "open": [100.0 + i for i in range(24)],
        "high": [101.0 + i for i in range(24)],
        "low": [99.0 + i for i in range(24)],
        "close": [100.5 + i for i in range(24)],
        "volume": [1000000.0 for _ in range(24)],
    }, index=timestamps)
    
    # Act
    resampled = await feature_service.resample_ohlcv(df, "4h")
    
    # Assert
    assert len(resampled) == 6  # 24 hours / 4 hours = 6 bars
    assert list(resampled.columns) == ["open", "high", "low", "close", "volume"]
    # Check aggregation: volume should be summed
    assert resampled["volume"].iloc[0] == 4000000.0  # 4 hours * 1M


@pytest.mark.asyncio
async def test_align_news_scores(feature_service, mock_analysis_repo):
    """Test news score alignment."""
    # Arrange
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]
    
    df = pd.DataFrame({
        "open": [100.0 for _ in range(10)],
        "high": [101.0 for _ in range(10)],
        "low": [99.0 for _ in range(10)],
        "close": [100.5 for _ in range(10)],
        "volume": [1000000.0 for _ in range(10)],
    }, index=timestamps)
    
    # Mock news scores at specific times
    news_scores = [
        (base_time + timedelta(hours=2), 0.5),
        (base_time + timedelta(hours=5), -0.3),
        (base_time + timedelta(hours=8), 0.7),
    ]
    mock_analysis_repo.get_news_scores_by_time_range.return_value = news_scores
    
    # Act
    aligned = await feature_service.align_news_scores(df, "AAPL")
    
    # Assert
    assert "news_score" in aligned.columns
    assert len(aligned) == 10
    # Check forward-fill behavior
    assert aligned["news_score"].iloc[0] == 0.0  # Before first news
    assert aligned["news_score"].iloc[2] == 0.5  # At first news
    assert aligned["news_score"].iloc[4] == 0.5  # Between first and second
    assert aligned["news_score"].iloc[5] == -0.3  # At second news


@pytest.mark.asyncio
async def test_align_news_scores_no_news(feature_service, mock_analysis_repo):
    """Test news score alignment with no news."""
    # Arrange
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]
    
    df = pd.DataFrame({
        "open": [100.0 for _ in range(10)],
        "high": [101.0 for _ in range(10)],
        "low": [99.0 for _ in range(10)],
        "close": [100.5 for _ in range(10)],
        "volume": [1000000.0 for _ in range(10)],
    }, index=timestamps)
    
    mock_analysis_repo.get_news_scores_by_time_range.return_value = []
    
    # Act
    aligned = await feature_service.align_news_scores(df, "AAPL")
    
    # Assert
    assert "news_score" in aligned.columns
    assert (aligned["news_score"] == 0.0).all()


@pytest.mark.asyncio
async def test_normalize_features_standard(feature_service, mock_scaler_repo):
    """Test standard normalization."""
    # Arrange
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]
    
    df = pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        "volume": [1000000.0 for _ in range(10)],
        "news_score": [0.0 for _ in range(10)],
    }, index=timestamps)
    
    # Mock no stored parameters
    mock_scaler_repo.get_params.return_value = None
    mock_scaler_repo.save_params.return_value = "test-id"
    
    # Act
    normalized_array, feature_names = await feature_service.normalize_features(df, "AAPL")
    
    # Assert
    assert normalized_array.shape == (10, 6)
    assert feature_names == ["open", "high", "low", "close", "volume", "news_score"]
    # Check that mean is approximately 0 and std is approximately 1
    assert np.abs(normalized_array[:, 0].mean()) < 0.1  # Close to 0
    assert np.abs(normalized_array[:, 0].std() - 1.0) < 0.1  # Close to 1


@pytest.mark.asyncio
async def test_normalize_features_with_stored_params(feature_service, mock_scaler_repo):
    """Test normalization with stored parameters."""
    # Arrange
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]
    
    df = pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        "volume": [1000000.0 for _ in range(10)],
        "news_score": [0.0 for _ in range(10)],
    }, index=timestamps)
    
    # Mock stored parameters
    def get_params_side_effect(symbol, feature_name):
        return ScalerParams(
            id=str(uuid.uuid4()),
            symbol=symbol,
            feature_name=feature_name,
            mean=100.0,
            std=10.0,
            method="standard",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
    mock_scaler_repo.get_params.side_effect = get_params_side_effect
    
    # Act
    normalized_array, feature_names = await feature_service.normalize_features(df, "AAPL")
    
    # Assert
    assert normalized_array.shape == (10, 6)
    assert feature_names == ["open", "high", "low", "close", "volume", "news_score"]
    # Verify stored params were used
    assert mock_scaler_repo.get_params.call_count == 6


@pytest.mark.asyncio
async def test_build_sequence_success(
    feature_service,
    mock_ohlcv_repo,
    mock_analysis_repo,
    mock_scaler_repo,
    sample_ohlcv_data,
):
    """Test successful feature sequence building."""
    # Arrange
    symbol = "AAPL"
    sequence_length = 168
    end_time = datetime(2024, 1, 10, 0, 0, 0)
    
    mock_ohlcv_repo.get_range.return_value = sample_ohlcv_data
    mock_analysis_repo.get_news_scores_by_time_range.return_value = [
        (datetime(2024, 1, 5, 0, 0, 0), 0.5),
    ]
    mock_scaler_repo.get_params.return_value = None
    mock_scaler_repo.save_params.return_value = "test-id"
    
    # Act
    feature_sequence = await feature_service.build_sequence(
        symbol=symbol,
        sequence_length=sequence_length,
        end_time=end_time,
    )
    
    # Assert
    assert feature_sequence.symbol == symbol
    assert feature_sequence.sequence.shape[0] == sequence_length
    assert feature_sequence.sequence.shape[1] == 6  # 6 features
    assert len(feature_sequence.feature_names) == 6
    assert len(feature_sequence.timestamps) == sequence_length
    assert "sequence_length" in feature_sequence.metadata


@pytest.mark.asyncio
async def test_build_sequence_no_data(
    feature_service,
    mock_ohlcv_repo,
    mock_analysis_repo,
    mock_scaler_repo,
):
    """Test feature sequence building with no OHLCV data."""
    # Arrange
    symbol = "AAPL"
    sequence_length = 168
    end_time = datetime(2024, 1, 10, 0, 0, 0)
    
    mock_ohlcv_repo.get_range.return_value = []
    
    # Act & Assert
    with pytest.raises(FeatureEngineeringError, match="No OHLCV data found"):
        await feature_service.build_sequence(
            symbol=symbol,
            sequence_length=sequence_length,
            end_time=end_time,
        )


@pytest.mark.asyncio
async def test_parse_frequency_to_hours(feature_service):
    """Test frequency parsing."""
    assert feature_service._parse_frequency_to_hours("1h") == 1.0
    assert feature_service._parse_frequency_to_hours("4h") == 4.0
    assert feature_service._parse_frequency_to_hours("1d") == 24.0
    assert feature_service._parse_frequency_to_hours("1w") == 168.0
    assert feature_service._parse_frequency_to_hours("1m") == 720.0
    # Test uppercase compatibility
    assert feature_service._parse_frequency_to_hours("1H") == 1.0
    assert feature_service._parse_frequency_to_hours("4H") == 4.0
