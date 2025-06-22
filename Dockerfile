# Multi-stage build để tối ưu kích thước
FROM python:3.10.16-slim-bullseye as builder

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libhdf5-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip và cài đặt wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements và cài đặt packages
COPY requirements-prod.txt .

# Cài đặt packages theo batch để tránh conflict
RUN pip install --no-cache-dir --timeout 1000 \
    numpy==2.2.5 \
    scipy==1.15.2 \
    scikit-learn==1.6.1

# Cài torch CPU version (quan trọng: không dùng CUDA cho deployment)
RUN pip install --no-cache-dir --timeout 1000 \
    torch==2.5.1+cpu \
    torchaudio==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Cài các packages AI/ML
RUN pip install --no-cache-dir --timeout 1000 \
    transformers==4.51.3 \
    sentence-transformers==4.1.0 \
    tokenizers==0.21.1

# Cài chromadb và dependencies
RUN pip install --no-cache-dir --timeout 1000 \
    chromadb==0.6.3 \
    chroma-hnswlib==0.7.6

# Cài các packages khác
RUN pip install --no-cache-dir --timeout 1000 -r requirements-prod.txt

# Production stage
FROM python:3.10.16-slim-bullseye

# Cài runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libhdf5-103 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment từ builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Tạo user non-root
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy source code
COPY --chown=app:app . .

# Tạo thư mục cần thiết
RUN mkdir -p chroma_db data benchmark logs

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/status || exit 1

# Command để chạy app
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]