# ChromaDB Setup Guide

ChromaDB can run in two modes for this application:

## Option 1: Persistent Local Mode (Recommended - No Docker Needed) ✅

This is the **easiest option** and works like SQLite - no server needed!

### Setup

1. **Already configured!** The system uses persistent local storage by default.

2. **Data location**: ChromaDB will store data in `./chroma_data/` directory

3. **Start the server**:
   ```bash
   ./run_server.sh
   ```

That's it! ChromaDB will automatically create and manage the local database.

### Configuration

The default settings in `.env` (or environment variables):
```bash
CHROMADB_USE_PERSISTENT=true
CHROMADB_PERSIST_DIRECTORY=./chroma_data
```

### Advantages
- ✅ No Docker required
- ✅ No separate server to manage
- ✅ Works like SQLite - just files on disk
- ✅ Perfect for development and single-machine deployments
- ✅ Automatic setup - no manual steps

---

## Option 2: Client-Server Mode (Docker)

Use this if you want ChromaDB running as a separate service (e.g., for production or multi-machine setups).

### Setup with Docker

1. **Create docker-compose.yml**:
   ```yaml
   version: '3.8'
   
   services:
     chromadb:
       image: chromadb/chroma:latest
       ports:
         - "8001:8000"
       volumes:
         - ./chroma_data:/chroma/chroma
       environment:
         - IS_PERSISTENT=TRUE
         - ANONYMIZED_TELEMETRY=FALSE
       restart: unless-stopped
   ```

2. **Start ChromaDB**:
   ```bash
   docker-compose up -d chromadb
   ```

3. **Update .env** to use client-server mode:
   ```bash
   CHROMADB_USE_PERSISTENT=false
   CHROMADB_HOST=localhost
   CHROMADB_PORT=8001
   ```

4. **Start your application**:
   ```bash
   ./run_server.sh
   ```

### Docker Commands

```bash
# Start ChromaDB
docker-compose up -d chromadb

# Check status
docker-compose ps

# View logs
docker-compose logs -f chromadb

# Stop ChromaDB
docker-compose down

# Stop and remove data
docker-compose down -v
```

### Alternative: Docker Run (without docker-compose)

```bash
# Start ChromaDB
docker run -d \
  --name chromadb \
  -p 8001:8000 \
  -v $(pwd)/chroma_data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma:latest

# Stop ChromaDB
docker stop chromadb

# Remove container
docker rm chromadb
```

---

## Switching Between Modes

### Switch to Persistent Local Mode (No Docker)
```bash
# In .env or environment
CHROMADB_USE_PERSISTENT=true
```

### Switch to Client-Server Mode (Docker)
```bash
# In .env or environment
CHROMADB_USE_PERSISTENT=false
CHROMADB_HOST=localhost
CHROMADB_PORT=8001
```

Then restart your application.

---

## Testing ChromaDB

### Check Health
```bash
# Start your server
./run_server.sh

# In another terminal, check health
curl http://localhost:8000/api/v1/health/chromadb
```

Expected response:
```json
{
  "status": "healthy",
  "chromadb_connected": true,
  "collection": "financial_articles"
}
```

### Test News Ingestion (uses ChromaDB)
```bash
curl -X POST "http://localhost:8000/api/v1/ingest-news" \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["yahoo_finance"],
    "symbols": ["AAPL"],
    "max_articles": 5
  }'
```

---

## Troubleshooting

### Persistent Mode Issues

**Problem**: Permission errors with `./chroma_data/`
```bash
# Fix permissions
chmod -R 755 ./chroma_data
```

**Problem**: Corrupted database
```bash
# Remove and recreate
rm -rf ./chroma_data
# Restart server - it will recreate automatically
```

### Client-Server Mode Issues

**Problem**: "Could not connect to a Chroma server"
```bash
# Check if Docker container is running
docker ps | grep chroma

# If not running, start it
docker-compose up -d chromadb

# Check logs
docker-compose logs chromadb
```

**Problem**: Port 8001 already in use
```bash
# Find what's using the port
lsof -i :8001

# Either stop that process or change ChromaDB port in docker-compose.yml
```

---

## Data Management

### Backup Data
```bash
# Persistent mode
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_data/

# Docker mode (data is in same location)
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_data/
```

### Restore Data
```bash
# Stop server/docker
# Extract backup
tar -xzf chroma_backup_YYYYMMDD.tar.gz
# Restart server/docker
```

### Clear All Data
```bash
# Stop server/docker first
rm -rf chroma_data/
# Restart - will create fresh database
```

---

## Recommendations

- **Development**: Use **Persistent Local Mode** (default) - simplest setup
- **Production (single machine)**: Use **Persistent Local Mode** - no extra complexity
- **Production (distributed)**: Use **Client-Server Mode** with Docker
- **Testing**: Use **Persistent Local Mode** - fast and isolated

---

## Current Configuration

Check your current mode:
```bash
# Check .env file
cat .env | grep CHROMADB

# Or check at runtime
curl http://localhost:8000/api/v1/health/chromadb
```

The system is currently configured for **Persistent Local Mode** by default.
