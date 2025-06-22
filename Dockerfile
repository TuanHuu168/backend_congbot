# Simplified Dockerfile for Railway deployment
FROM python:3.10.16-slim-bullseye

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip và cài đặt wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Tạo user non-root
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements và cài đặt packages
COPY --chown=app:app requirements-prod.txt .

# Cài đặt packages một lần từ requirements
RUN pip install --no-cache-dir --timeout 1000 -r requirements-prod.txt

# Copy source code
COPY --chown=app:app . .

# Copy and make start script executable
COPY --chown=app:app start.sh .
RUN chmod +x start.sh

# Tạo thư mục cần thiết
RUN mkdir -p chroma_db data benchmark logs

# Expose port (Railway will set PORT env var)
EXPOSE 8001

# Use start script
CMD ["./start.sh"]