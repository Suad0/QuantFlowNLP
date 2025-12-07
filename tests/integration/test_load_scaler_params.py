"""Integration tests for load_scaler_params script.

Tests the functionality of loading pre-trained scaler parameters from JSON
into the database.
"""

import json
import pytest
import tempfile
from pathlib import Path

from app.adapters.database import DatabaseManager
from app.repositories.scaler_repository import SQLiteScalerRepository


@pytest.mark.asyncio
async def test_load_scaler_params_integration():
    """Test loading scaler parameters from JSON file into database."""
    # Create a temporary scaler params file
    scaler_data = {
        "mean": [100.0, 105.0, 95.0, 100.0, 1000000.0, 0.0],
        "scale": [10.0, 12.0, 8.0, 10.0, 500000.0, 1.0],
        "cols": ["open", "high", "low", "close", "volume", "news_score"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(scaler_data, f)
        temp_file = Path(f.name)
    
    try:
        # Import the load function
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.load_scaler_params import load_scaler_params
        
        # Load parameters for test symbol
        test_symbol = "TEST"
        await load_scaler_params(test_symbol, temp_file)
        
        # Verify parameters were loaded
        db = DatabaseManager()
        await db.connect()
        conn = await db.get_connection()
        repo = SQLiteScalerRepository(conn)
        
        params = await repo.get_all_params(test_symbol)
        assert len(params) == 6
        
        # Verify each feature
        feature_map = {p.feature_name: p for p in params}
        
        assert "open" in feature_map
        assert feature_map["open"].mean == 100.0
        assert feature_map["open"].std == 10.0
        
        assert "high" in feature_map
        assert feature_map["high"].mean == 105.0
        assert feature_map["high"].std == 12.0
        
        assert "low" in feature_map
        assert feature_map["low"].mean == 95.0
        assert feature_map["low"].std == 8.0
        
        assert "close" in feature_map
        assert feature_map["close"].mean == 100.0
        assert feature_map["close"].std == 10.0
        
        assert "volume" in feature_map
        assert feature_map["volume"].mean == 1000000.0
        assert feature_map["volume"].std == 500000.0
        
        assert "news_score" in feature_map
        assert feature_map["news_score"].mean == 0.0
        assert feature_map["news_score"].std == 1.0
        
        # Verify method is standard
        for param in params:
            assert param.method == "standard"
        
        await db.disconnect()
        
    finally:
        # Clean up temp file
        temp_file.unlink()


@pytest.mark.asyncio
async def test_load_scaler_params_update_existing():
    """Test updating existing scaler parameters."""
    # Create initial parameters
    scaler_data_v1 = {
        "mean": [100.0, 105.0, 95.0, 100.0, 1000000.0, 0.0],
        "scale": [10.0, 12.0, 8.0, 10.0, 500000.0, 1.0],
        "cols": ["open", "high", "low", "close", "volume", "news_score"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(scaler_data_v1, f)
        temp_file = Path(f.name)
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.load_scaler_params import load_scaler_params
        
        test_symbol = "UPDATE_TEST"
        
        # Load initial parameters
        await load_scaler_params(test_symbol, temp_file)
        
        # Update the file with new values
        scaler_data_v2 = {
            "mean": [110.0, 115.0, 105.0, 110.0, 2000000.0, 0.5],
            "scale": [15.0, 17.0, 13.0, 15.0, 750000.0, 1.5],
            "cols": ["open", "high", "low", "close", "volume", "news_score"]
        }
        
        with open(temp_file, 'w') as f:
            json.dump(scaler_data_v2, f)
        
        # Load updated parameters
        await load_scaler_params(test_symbol, temp_file)
        
        # Verify parameters were updated
        db = DatabaseManager()
        await db.connect()
        conn = await db.get_connection()
        repo = SQLiteScalerRepository(conn)
        
        params = await repo.get_all_params(test_symbol)
        assert len(params) == 6
        
        # Verify updated values
        feature_map = {p.feature_name: p for p in params}
        
        assert feature_map["open"].mean == 110.0
        assert feature_map["open"].std == 15.0
        
        assert feature_map["news_score"].mean == 0.5
        assert feature_map["news_score"].std == 1.5
        
        await db.disconnect()
        
    finally:
        temp_file.unlink()


@pytest.mark.asyncio
async def test_load_scaler_params_file_not_found():
    """Test error handling when scaler file doesn't exist."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.load_scaler_params import load_scaler_params
    
    non_existent_file = Path("/tmp/non_existent_scaler.json")
    
    with pytest.raises(FileNotFoundError):
        await load_scaler_params("TEST", non_existent_file)


@pytest.mark.asyncio
async def test_load_scaler_params_invalid_json():
    """Test error handling with invalid JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json content {{{")
        temp_file = Path(f.name)
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.load_scaler_params import load_scaler_params
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            await load_scaler_params("TEST", temp_file)
            
    finally:
        temp_file.unlink()


@pytest.mark.asyncio
async def test_load_scaler_params_missing_fields():
    """Test error handling when required fields are missing."""
    # Missing 'cols' field
    scaler_data = {
        "mean": [100.0, 105.0],
        "scale": [10.0, 12.0]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(scaler_data, f)
        temp_file = Path(f.name)
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.load_scaler_params import load_scaler_params
        
        with pytest.raises(ValueError, match="Missing required field"):
            await load_scaler_params("TEST", temp_file)
            
    finally:
        temp_file.unlink()
