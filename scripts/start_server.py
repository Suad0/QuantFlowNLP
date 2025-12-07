#!/usr/bin/env python3
"""Start the Quantitative Trading Intelligence System API server.

This script starts the FastAPI application with uvicorn.

Usage:
    uv run python scripts/start_server.py
    
Or with custom settings:
    uv run python scripts/start_server.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import sys

import uvicorn


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(
        description="Start the Quantitative Trading Intelligence System API"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log level (default: info)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  QUANTITATIVE TRADING INTELLIGENCE SYSTEM")
    print("=" * 80)
    print(f"\n🚀 Starting API server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Reload: {args.reload}")
    print(f"   Log Level: {args.log_level}")
    print(f"\n📚 Documentation:")
    print(f"   Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"   ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"   OpenAPI: http://{args.host}:{args.port}/openapi.json")
    print(f"\n💡 Tips:")
    print(f"   - Press Ctrl+C to stop the server")
    print(f"   - Use --reload for development")
    print(f"   - Check /api/v1/health for system status")
    print("\n" + "=" * 80 + "\n")
    
    try:
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n\n❌ Error starting server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
