#!/bin/bash
# Startup script for Railway deployment

echo "Starting RespireX API..."

# Set environment variables
export PYTHONPATH=/app
export PORT=${PORT:-8000}

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port $PORT
