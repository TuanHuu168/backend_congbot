#!/bin/bash

# Railway startup script

# Set environment variables
export PYTHONPATH=/home/app:$PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""  # Force CPU only

# Railway sets PORT environment variable
export PORT=${PORT:-8001}

# Create necessary directories
mkdir -p /home/app/chroma_db
mkdir -p /home/app/data
mkdir -p /home/app/benchmark/results
mkdir -p /home/app/logs

# Initialize MongoDB indexes (if MongoDB is available)
echo "Initializing database..."
python -c "
try:
    from database.mongodb_client import mongodb_client
    mongodb_client.create_indexes()
    print('Database indexes created successfully')
except Exception as e:
    print(f'Database initialization error (will retry later): {e}')
" || echo "Database not ready, will initialize during runtime"

# Start the application
echo "Starting FastAPI application on port $PORT..."
exec python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --access-log