"""Feature engineering service for preparing time-series data.

This module provides the FeatureEngineeringService which prepares feature sequences
for xLSTM model inference by fetching OHLCV data, aligning news scores, and
normalizing features.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.exceptions import FeatureEngineeringError
from app.models.domain import FeatureSequence, ScalerParams
from app.repositories.base import AnalysisRepository, OHLCVRepository, ScalerRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineeringService:
    """Service for building feature sequences for model inference.
    
    This service orchestrates the process of:
    1. Fetching OHLCV data for a symbol
    2. Resampling to target frequency
    3. Aligning news scores with timestamps
    4. Normalizing features using stored parameters
    5. Returning properly shaped feature matrix
    """

    def __init__(
        self,
        ohlcv_repository: OHLCVRepository,
        analysis_repository: AnalysisRepository,
        scaler_repository: ScalerRepository,
    ):
        """Initialize service with required repositories.
        
        Args:
            ohlcv_repository: Repository for OHLCV data
            analysis_repository: Repository for article analysis
            scaler_repository: Repository for scaler parameters
        """
        self.ohlcv_repo = ohlcv_repository
        self.analysis_repo = analysis_repository
        self.scaler_repo = scaler_repository

    async def build_sequence(
        self,
        symbol: str,
        sequence_length: Optional[int] = None,
        end_time: Optional[datetime] = None,
    ) -> FeatureSequence:
        """Build feature sequence for model inference.
        
        This is the main orchestration method that coordinates all steps
        of feature preparation.
        
        Args:
            symbol: Trading symbol
            sequence_length: Length of sequence (default from config)
            end_time: End time for sequence (default: now)
            
        Returns:
            FeatureSequence ready for model inference
            
        Raises:
            FeatureEngineeringError: If feature building fails
        """
        try:
            # Use defaults from config if not provided
            if sequence_length is None:
                sequence_length = settings.feature_default_sequence_length
            if end_time is None:
                end_time = datetime.utcnow()
            
            logger.info(
                f"Building feature sequence for {symbol}, "
                f"length={sequence_length}, end_time={end_time}"
            )
            
            # Step 1: Get OHLCV data
            ohlcv_df = await self.get_ohlcv_data(symbol, sequence_length, end_time)
            
            if ohlcv_df.empty:
                raise FeatureEngineeringError(
                    f"No OHLCV data found for {symbol} in the specified time range"
                )
            
            # Step 2: Resample to target frequency
            resampled_df = await self.resample_ohlcv(
                ohlcv_df,
                settings.feature_target_frequency,
            )
            
            # Step 3: Align news scores
            aligned_df = await self.align_news_scores(resampled_df, symbol)
            
            # Step 4: Normalize features
            normalized_array, feature_names = await self.normalize_features(
                aligned_df,
                symbol,
            )
            
            # Ensure we have exactly sequence_length rows
            if len(normalized_array) < sequence_length:
                raise FeatureEngineeringError(
                    f"Insufficient data: got {len(normalized_array)} rows, "
                    f"need {sequence_length}"
                )
            
            # Take the last sequence_length rows
            normalized_array = normalized_array[-sequence_length:]
            timestamps = aligned_df.index[-sequence_length:].to_pydatetime().tolist()
            
            logger.info(
                f"Successfully built feature sequence for {symbol}: "
                f"shape={normalized_array.shape}"
            )
            
            return FeatureSequence(
                symbol=symbol,
                sequence=normalized_array,
                feature_names=feature_names,
                timestamps=timestamps,
                metadata={
                    "sequence_length": sequence_length,
                    "end_time": end_time.isoformat(),
                    "target_frequency": settings.feature_target_frequency,
                    "normalization_method": settings.feature_normalization_method,
                },
            )
            
        except FeatureEngineeringError:
            raise
        except Exception as e:
            logger.error(f"Failed to build feature sequence for {symbol}: {e}")
            raise FeatureEngineeringError(
                f"Failed to build feature sequence: {e}"
            ) from e

    async def get_ohlcv_data(
        self,
        symbol: str,
        sequence_length: int,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Get OHLCV data for the specified symbol and time range.
        
        Calculates the start time based on sequence length and target frequency,
        then fetches data from the repository.
        
        Args:
            symbol: Trading symbol
            sequence_length: Desired sequence length
            end_time: End time for data
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
            
        Raises:
            FeatureEngineeringError: If data fetching fails
        """
        try:
            # Calculate start time based on sequence length and frequency
            # Add buffer to ensure we have enough data after resampling
            freq_hours = self._parse_frequency_to_hours(
                settings.feature_target_frequency
            )
            buffer_multiplier = 1.5  # 50% buffer
            hours_needed = int(sequence_length * freq_hours * buffer_multiplier)
            start_time = end_time - timedelta(hours=hours_needed)
            
            logger.debug(
                f"Fetching OHLCV data for {symbol} from {start_time} to {end_time}"
            )
            
            # Fetch data from repository
            ohlcv_data = await self.ohlcv_repo.get_range(symbol, start_time, end_time)
            
            if not ohlcv_data:
                logger.warning(f"No OHLCV data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data_dicts = [
                {
                    "timestamp": d.timestamp,
                    "open": d.open,
                    "high": d.high,
                    "low": d.low,
                    "close": d.close,
                    "volume": d.volume,
                }
                for d in ohlcv_data
            ]
            
            df = pd.DataFrame(data_dicts)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            
            logger.debug(f"Fetched {len(df)} OHLCV records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            raise FeatureEngineeringError(f"Failed to get OHLCV data: {e}") from e

    async def resample_ohlcv(
        self,
        df: pd.DataFrame,
        target_freq: str,
    ) -> pd.DataFrame:
        """Resample OHLCV data to target frequency.
        
        Uses pandas resampling with appropriate aggregation functions:
        - open: first
        - high: max
        - low: min
        - close: last
        - volume: sum
        
        Args:
            df: DataFrame with OHLCV data
            target_freq: Target frequency (e.g., '1H', '4H', '1D')
            
        Returns:
            Resampled DataFrame
            
        Raises:
            FeatureEngineeringError: If resampling fails
        """
        try:
            if df.empty:
                return df
            
            logger.debug(f"Resampling OHLCV data to {target_freq}")
            
            # Resample with appropriate aggregation
            resampled = df.resample(target_freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            
            # Drop rows with NaN values (incomplete bars)
            resampled.dropna(inplace=True)
            
            logger.debug(
                f"Resampled from {len(df)} to {len(resampled)} records"
            )
            
            return resampled
            
        except Exception as e:
            logger.error(f"Failed to resample OHLCV data: {e}")
            raise FeatureEngineeringError(f"Failed to resample OHLCV: {e}") from e

    async def align_news_scores(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Align news scores with OHLCV timestamps using forward-fill.
        
        Fetches news scores for the time range covered by the DataFrame,
        then aligns them with OHLCV timestamps using forward-fill logic
        (news score persists until next news event).
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol (currently unused, but kept for future filtering)
            
        Returns:
            DataFrame with news_score column added
            
        Raises:
            FeatureEngineeringError: If alignment fails
        """
        try:
            if df.empty:
                return df
            
            # Get time range from DataFrame
            start_time = df.index.min().to_pydatetime()
            end_time = df.index.max().to_pydatetime()
            
            logger.debug(
                f"Aligning news scores for time range {start_time} to {end_time}"
            )
            
            # Fetch news scores from repository
            news_scores = await self.analysis_repo.get_news_scores_by_time_range(
                start_time,
                end_time,
            )
            
            if not news_scores:
                logger.warning("No news scores found, using default value 0.0")
                df["news_score"] = 0.0
                return df
            
            # Create Series from news scores
            news_series = pd.Series(
                data=[score for _, score in news_scores],
                index=[ts for ts, _ in news_scores],
                name="news_score",
            )
            
            # Combine with OHLCV data and forward-fill
            df = df.copy()
            df["news_score"] = news_series
            df["news_score"] = df["news_score"].ffill()
            
            # Fill any remaining NaN with 0.0 (before first news event)
            df["news_score"] = df["news_score"].fillna(0.0)
            
            logger.debug(
                f"Aligned {len(news_scores)} news scores with OHLCV data"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to align news scores: {e}")
            raise FeatureEngineeringError(f"Failed to align news scores: {e}") from e

    async def normalize_features(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> tuple[np.ndarray, List[str]]:
        """Normalize features using stored scaler parameters.
        
        Applies normalization based on the configured method:
        - standard: (x - mean) / std
        - minmax: (x - min) / (max - min)
        - robust: Uses median and IQR (not yet implemented)
        
        Args:
            df: DataFrame with features
            symbol: Trading symbol
            
        Returns:
            Tuple of (normalized array, feature names)
            
        Raises:
            FeatureEngineeringError: If normalization fails
        """
        try:
            if df.empty:
                return np.array([]), []
            
            # Get feature columns from config
            feature_cols = settings.feature_columns
            
            # Ensure all required columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                raise FeatureEngineeringError(
                    f"Missing required columns: {missing_cols}"
                )
            
            # Select only the feature columns in the specified order
            feature_df = df[feature_cols].copy()
            
            logger.debug(
                f"Normalizing {len(feature_cols)} features using "
                f"{settings.feature_normalization_method} method"
            )
            
            # Get normalization method
            method = settings.feature_normalization_method
            
            if method == "standard":
                normalized_df = await self._normalize_standard(feature_df, symbol)
            elif method == "minmax":
                normalized_df = await self._normalize_minmax(feature_df, symbol)
            elif method == "robust":
                normalized_df = await self._normalize_robust(feature_df, symbol)
            else:
                raise FeatureEngineeringError(
                    f"Unknown normalization method: {method}"
                )
            
            # Convert to numpy array
            normalized_array = normalized_df.values
            
            logger.debug(
                f"Normalized features: shape={normalized_array.shape}"
            )
            
            return normalized_array, feature_cols
            
        except FeatureEngineeringError:
            raise
        except Exception as e:
            logger.error(f"Failed to normalize features: {e}")
            raise FeatureEngineeringError(f"Failed to normalize features: {e}") from e

    async def _normalize_standard(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Apply standard normalization (z-score).
        
        Args:
            df: DataFrame with features
            symbol: Trading symbol
            
        Returns:
            Normalized DataFrame
        """
        normalized_df = df.copy()
        
        for col in df.columns:
            # Try to get stored parameters
            params = await self.scaler_repo.get_params(symbol, col)
            
            if params and params.method == "standard":
                # Use stored parameters
                mean = params.mean
                std = params.std
                logger.debug(
                    f"Using stored params for {col}: mean={mean:.4f}, std={std:.4f}"
                )
            else:
                # Calculate from data
                mean = df[col].mean()
                std = df[col].std()
                
                # Store parameters for future use
                params = ScalerParams(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    feature_name=col,
                    mean=float(mean),
                    std=float(std),
                    method="standard",
                    created_at=datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow(),
                    updated_at=datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow(),
                )
                await self.scaler_repo.save_params(params)
                logger.debug(
                    f"Calculated and stored params for {col}: "
                    f"mean={mean:.4f}, std={std:.4f}"
                )
            
            # Apply normalization
            if std > 0:
                normalized_df[col] = (df[col] - mean) / std
            else:
                # If std is 0, just center the data
                normalized_df[col] = df[col] - mean
        
        return normalized_df

    async def _normalize_minmax(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Apply min-max normalization.
        
        Args:
            df: DataFrame with features
            symbol: Trading symbol
            
        Returns:
            Normalized DataFrame
        """
        normalized_df = df.copy()
        
        for col in df.columns:
            # Try to get stored parameters
            params = await self.scaler_repo.get_params(symbol, col)
            
            if params and params.method == "minmax" and params.min is not None and params.max is not None:
                # Use stored parameters
                min_val = params.min
                max_val = params.max
                logger.debug(
                    f"Using stored params for {col}: min={min_val:.4f}, max={max_val:.4f}"
                )
            else:
                # Calculate from data
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Store parameters for future use
                params = ScalerParams(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    feature_name=col,
                    mean=0.0,  # Not used for minmax
                    std=1.0,   # Not used for minmax
                    min=float(min_val),
                    max=float(max_val),
                    method="minmax",
                    created_at=datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow(),
                    updated_at=datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow(),
                )
                await self.scaler_repo.save_params(params)
                logger.debug(
                    f"Calculated and stored params for {col}: "
                    f"min={min_val:.4f}, max={max_val:.4f}"
                )
            
            # Apply normalization
            if max_val > min_val:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                # If range is 0, set to 0.5 (middle of [0, 1])
                normalized_df[col] = 0.5
        
        return normalized_df

    async def _normalize_robust(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Apply robust normalization using median and IQR.
        
        Args:
            df: DataFrame with features
            symbol: Trading symbol
            
        Returns:
            Normalized DataFrame
        """
        # For now, fall back to standard normalization
        # TODO: Implement robust normalization with median and IQR
        logger.warning("Robust normalization not yet implemented, using standard")
        return await self._normalize_standard(df, symbol)

    def _parse_frequency_to_hours(self, freq: str) -> float:
        """Parse frequency string to hours.
        
        Args:
            freq: Frequency string (e.g., '1h', '4h', '1D')
            
        Returns:
            Number of hours
        """
        # Normalize to lowercase for pandas compatibility
        freq_lower = freq.lower()
        
        if freq_lower.endswith("h"):
            return float(freq_lower[:-1])
        elif freq_lower.endswith("d"):
            return float(freq_lower[:-1]) * 24
        elif freq_lower.endswith("w"):
            return float(freq_lower[:-1]) * 24 * 7
        elif freq_lower.endswith("m"):
            # Approximate month as 30 days
            return float(freq_lower[:-1]) * 24 * 30
        else:
            # Default to 1 hour
            logger.warning(f"Unknown frequency format: {freq}, defaulting to 1 hour")
            return 1.0
