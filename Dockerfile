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

# Step 1: Cài base packages trước
RUN pip install --no-cache-dir --timeout 1000 \
    numpy==2.2.5 \
    pydantic==2.11.3 \
    fastapi==0.115.9 \
    uvicorn==0.34.2

# Step 2: Cài PyTorch CPU-only
RUN pip install --no-cache-dir --timeout 1000 \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 3: Cài database packages
RUN pip install --no-cache-dir --timeout 1000 \
    pymongo==4.13.0

# Step 4: Cài AI packages cơ bản
RUN pip install --no-cache-dir --timeout 1000 \
    transformers==4.44.2 \
    sentence-transformers==3.0.1 \
    google-generativeai==0.8.5

# Step 5: Cài utilities
RUN pip install --no-cache-dir --timeout 1000 \
    python-multipart==0.0.20 \
    python-dotenv==1.1.0 \
    bcrypt==4.3.0 \
    requests==2.32.3 \
    httpx==0.28.1 \
    click==8.1.8

# Step 6: LangChain minimal
RUN pip install --no-cache-dir --timeout 1000 \
    langchain-core==0.3.55 \
    langchain-google-genai==2.1.3 || echo "LangChain install failed, continuing..."

# Step 7: Optional packages (continue on error)
RUN pip install --no-cache-dir --timeout 1000 \
    openai==1.82.0 \
    email_validator==2.2.0 \
    scikit-learn==1.5.1 || echo "Optional packages install failed, continuing..."

# Copy source code
COPY --chown=app:app . .

# Copy and make start script executable
COPY --chown=app:app start.sh .
RUN chmod +x start.sh

# Tạo thư mục cần thiết
RUN mkdir -p chroma_db data benchmark logs

# Expose port
EXPOSE 8001

# Use start script
CMD ["./start.sh"]