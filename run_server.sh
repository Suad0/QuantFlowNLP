#!/bin/bash
# Simple script to start the API server

echo "================================================================================"
echo "  QUANTITATIVE TRADING INTELLIGENCE SYSTEM"
echo "================================================================================"
echo ""
echo "🚀 Starting API server on http://localhost:8000"
echo ""
echo "📚 Documentation:"
echo "   - Swagger UI: http://localhost:8000/docs"
echo "   - ReDoc: http://localhost:8000/redoc"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo ""
echo "================================================================================"
echo ""

# Start the server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
