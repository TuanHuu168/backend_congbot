# Railway Dockerfile với exact versions từ requirements.txt gốc
FROM python:3.10.16-slim-bullseye

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libhdf5-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory as root to install packages
WORKDIR /app

# Upgrade pip và tools
RUN pip install --no-cache-dir --upgrade pip==25.0 setuptools==75.8.0 wheel==0.45.1

# Copy requirements
COPY requirements-railway.txt .

# Step 1: Cài PyTorch CPU trước (quan trọng phải cài trước)
RUN pip install --no-cache-dir --timeout 1500 \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Cài packages từ requirements-railway.txt
RUN pip install --no-cache-dir --timeout 1500 -r requirements-railway.txt

# Step 3: Fix Google genai import issue - ensure correct version
RUN pip install --no-cache-dir --force-reinstall \
    google-genai==1.11.0 \
    google-generativeai==0.8.5

# Tạo non-root user và fix PATH
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /home/app/.local/bin \
    && chown -R app:app /home/app

# Switch to app user
USER app
WORKDIR /home/app

# Fix PATH and Python environment
ENV PATH="/home/app/.local/bin:/usr/local/bin:$PATH"
ENV PYTHONPATH="/home/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV USE_GPU=False

# Copy source code
COPY --chown=app:app . .

# Tạo thư mục cần thiết
RUN mkdir -p chroma_db data benchmark/results logs tmp

# Expose port
EXPOSE 8001

# Start command với timeout cao hơn cho Railway
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--timeout-keep-alive", "300", "--access-log"]