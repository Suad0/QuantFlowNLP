#!/usr/bin/env python3
"""Load pre-trained scaler parameters into database.

This script loads scaler parameters from the trained model's scaler_params.json
file and inserts them into the SQLite database. This ensures that feature
normalization during inference matches the model's training data.

The script reads parameters from model_downloaded/scaler_params.json and
inserts them for a specified trading symbol (default: AAPL, which was used
during training).

Usage:
    python scripts/load_scaler_params.py [--symbol SYMBOL]
    
Examples:
    # Load for default symbol (AAPL)
    python scripts/load_scaler_params.py
    
    # Load for specific symbol
    python scripts/load_scaler_params.py --symbol TSLA
"""

import asyncio
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.adapters.database import DatabaseManager
from app.core.config import settings
from app.models.domain.scaler import ScalerParams
from app.repositories.scaler_repository import SQLiteScalerRepository
from app.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def load_scaler_params(symbol: str, scaler_file: Path) -> None:
    """Load scaler parameters from JSON file into database.
    
    Args:
        symbol: Trading symbol to associate with the parameters
        scaler_file: Path to scaler_params.json file
        
    Raises:
        FileNotFoundError: If scaler file doesn't exist
        ValueError: If scaler file format is invalid
    """
    # Validate scaler file exists
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    
    logger.info(f"Loading scaler parameters from: {scaler_file}")
    
    # Load scaler parameters from JSON
    try:
        with open(scaler_file, 'r') as f:
            scaler_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in scaler file: {e}") from e
    
    # Validate required fields
    required_fields = ['mean', 'scale', 'cols']
    for field in required_fields:
        if field not in scaler_data:
            raise ValueError(f"Missing required field in scaler file: {field}")
    
    means = scaler_data['mean']
    scales = scaler_data['scale']
    feature_names = scaler_data['cols']
    
    # Validate data consistency
    if not (len(means) == len(scales) == len(feature_names)):
        raise ValueError(
            f"Inconsistent data lengths: means={len(means)}, "
            f"scales={len(scales)}, features={len(feature_names)}"
        )
    
    # Expected feature names for the xLSTM model
    expected_features = ['open', 'high', 'low', 'close', 'volume', 'news_score']
    if feature_names != expected_features:
        logger.warning(
            f"Feature names mismatch! Expected: {expected_features}, "
            f"Got: {feature_names}"
        )
        logger.warning("Proceeding anyway, but this may cause issues during inference")
    
    logger.info(f"Found {len(feature_names)} features: {feature_names}")
    logger.info(f"Inserting scaler parameters for symbol: {symbol}")
    
    # Connect to database
    db_manager = DatabaseManager()
    try:
        await db_manager.connect()
        conn = await db_manager.get_connection()
        
        # Create repository
        scaler_repo = SQLiteScalerRepository(conn)
        
        # Insert parameters for each feature
        now = datetime.now(timezone.utc)
        inserted_count = 0
        updated_count = 0
        
        for feature_name, mean, scale in zip(feature_names, means, scales):
            # Check if parameters already exist
            existing = await scaler_repo.get_params(symbol, feature_name)
            
            if existing:
                logger.info(
                    f"  - {feature_name}: Updating existing parameters "
                    f"(mean={mean:.4f}, std={scale:.4f})"
                )
                # Update existing parameters
                existing.mean = mean
                existing.std = scale
                existing.updated_at = now
                await scaler_repo.update_params(existing)
                updated_count += 1
            else:
                logger.info(
                    f"  - {feature_name}: Inserting new parameters "
                    f"(mean={mean:.4f}, std={scale:.4f})"
                )
                # Create new parameters
                params = ScalerParams(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    feature_name=feature_name,
                    mean=mean,
                    std=scale,
                    min=None,  # Not used for standard normalization
                    max=None,  # Not used for standard normalization
                    method='standard',  # The model uses standard normalization
                    created_at=now,
                    updated_at=now,
                )
                await scaler_repo.save_params(params)
                inserted_count += 1
        
        logger.info(
            f"Successfully loaded scaler parameters: "
            f"{inserted_count} inserted, {updated_count} updated"
        )
        
        # Verify insertion
        all_params = await scaler_repo.get_all_params(symbol)
        logger.info(f"Total scaler parameters for {symbol}: {len(all_params)}")
        
        # Display summary
        logger.info("\nScaler Parameters Summary:")
        logger.info(f"{'Feature':<15} {'Mean':<15} {'Std':<15}")
        logger.info("-" * 45)
        for param in sorted(all_params, key=lambda p: feature_names.index(p.feature_name) 
                           if p.feature_name in feature_names else 999):
            logger.info(f"{param.feature_name:<15} {param.mean:<15.4f} {param.std:<15.4f}")
        
    except Exception as e:
        logger.error(f"Failed to load scaler parameters: {e}")
        raise
    finally:
        await db_manager.disconnect()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load pre-trained scaler parameters into database"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Trading symbol to associate with parameters (default: AAPL)",
    )
    parser.add_argument(
        "--scaler-file",
        type=Path,
        default=Path("model_downloaded/scaler_params.json"),
        help="Path to scaler_params.json file (default: model_downloaded/scaler_params.json)",
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(load_scaler_params(args.symbol, args.scaler_file))
        logger.info("\n✓ Scaler parameters loaded successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Failed to load scaler parameters: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
