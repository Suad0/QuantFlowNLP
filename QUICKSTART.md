# Quick Start Guide

## Get Your API Running in 3 Steps

### Step 1: Start the Server

Choose one of these methods:

**Option A: Using the shell script (easiest)**
```bash
./run_server.sh
```

**Option B: Using uvicorn directly**
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Using the Python script**
```bash
uv run python scripts/start_server.py --reload
```

The server will start on **http://localhost:8000**

### Step 2: Test the API

Open your browser and visit:
- **Swagger UI**: http://localhost:8000/docs (interactive testing)
- **Health Check**: http://localhost:8000/api/v1/health

Or run the test script:
```bash
uv run python scripts/test_all_endpoints.py
```

### Step 3: Try Some Endpoints

**Check Health:**
```bash
curl http://localhost:8000/api/v1/health
```

**Price an Option:**
```bash
curl -X POST http://localhost:8000/api/v1/quant/pricing/option \
  -H "Content-Type: application/json" \
  -d '{
    "spot": 100.0,
    "strike": 105.0,
    "time_to_maturity": 1.0,
    "risk_free_rate": 0.05,
    "volatility": 0.2,
    "option_type": "call"
  }'
```

**Build Yield Curve:**
```bash
curl -X POST http://localhost:8000/api/v1/quant/yield-curve \
  -H "Content-Type: application/json" \
  -d '{
    "market_rates": [
      {"maturity": 0.25, "rate": 0.02, "instrument_type": "deposit"},
      {"maturity": 1.0, "rate": 0.03, "instrument_type": "swap"},
      {"maturity": 5.0, "rate": 0.04, "instrument_type": "bond"}
    ],
    "interpolation_method": "linear"
  }'
```

---

## What's Available

### ✅ Working Out of the Box

These endpoints work immediately without any setup:

1. **Health Checks** - Monitor system status
2. **Quantitative Finance** - Option pricing, bond pricing, yield curves
3. **Portfolio Optimization** - Optimize portfolios (needs historical data)
4. **API Documentation** - Interactive Swagger UI

### 🔧 Requires Setup

These features need additional services:

1. **News Ingestion** - Requires ChromaDB
   ```bash
   docker run -p 8001:8000 chromadb/chroma
   ```

2. **NLP Analysis** - Requires Ollama
   ```bash
   ollama serve
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

3. **Predictions** - Requires trained model
   ```bash
   # Model files should be in model_downloaded/
   curl -X POST http://localhost:8000/api/v1/model/load
   ```

---

## Common Commands

### Start Server
```bash
./run_server.sh
```

### Run Tests
```bash
uv run python scripts/test_all_endpoints.py
```

### Check Health
```bash
curl http://localhost:8000/api/v1/health
```

### View Documentation
```bash
open http://localhost:8000/docs
```

---

## Troubleshooting

### Port 8000 Already in Use
```bash
# Use a different port
uv run uvicorn app.main:app --port 8001
```

### Import Errors
```bash
# Reinstall dependencies
uv sync
```

### Database Errors
```bash
# Reinitialize database
rm data/trading_system.db
uv run python scripts/init_db.py
```

---

## Next Steps

1. ✅ **API is running** - You're all set!
2. 📖 **Read the docs** - Visit http://localhost:8000/docs
3. 🧪 **Run tests** - `uv run python scripts/test_all_endpoints.py`
4. 📚 **Learn more** - Check `TESTING_GUIDE.md` for detailed examples

---

## Need Help?

- **API Documentation**: http://localhost:8000/docs
- **Test Results**: See `API_TEST_RESULTS.md`
- **Detailed Guide**: See `TESTING_GUIDE.md`
- **Health Status**: http://localhost:8000/api/v1/health

---

**Status**: ✅ Ready to use!  
**Endpoints**: 20 available  
**Documentation**: Complete
