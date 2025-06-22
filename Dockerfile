# Brute force Dockerfile - ignore ALL conflicts
FROM python:3.10.16-slim-bullseye

# System deps
RUN apt-get update && apt-get install -y \
    build-essential curl git gcc g++ cmake pkg-config libhdf5-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Step 1: PyTorch CPU first
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Core packages - install with --force-reinstall --no-deps
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    numpy==2.2.5 \
    scipy==1.15.2 \
    pydantic==2.11.3 \
    pydantic_core==2.33.1 \
    fastapi==0.115.9 \
    uvicorn==0.34.3 \
    starlette==0.45.3

# Step 3: Database packages
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    pymongo==4.13.0 \
    chromadb==0.6.3 \
    chroma-hnswlib==0.7.6

# Step 4: Google AI packages - FORCE INSTALL exact versions
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    protobuf==5.29.4 \
    grpcio==1.71.0 \
    grpcio-status==1.71.0 \
    google-auth==2.39.0 \
    google-api-core==2.24.2 \
    google-ai-generativelanguage==0.6.17 \
    google-generativeai==0.8.5 \
    google-genai==1.11.0

# Step 5: ML packages
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    transformers==4.51.3 \
    sentence-transformers==4.1.0 \
    tokenizers==0.21.1 \
    huggingface-hub==0.30.2 \
    safetensors==0.5.3 \
    scikit-learn==1.6.1 \
    pandas==2.3.0

# Step 6: LangChain (optional - comment out if causes issues)
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    langchain==0.3.24 \
    langchain-core==0.3.55 \
    langchain-chroma==0.2.3 \
    langchain-community==0.3.22 \
    langchain-google-genai==2.1.3 \
    langchain-huggingface==0.1.2 \
    langchain-text-splitters==0.3.8 \
    langsmith==0.3.33 || echo "LangChain failed, continuing..."

# Step 7: Web utilities
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    httpx==0.28.1 \
    httpcore==1.0.8 \
    aiohttp==3.11.18 \
    requests==2.32.3 \
    urllib3==2.4.0

# Step 8: Basic utilities
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    python-multipart==0.0.20 \
    python-dotenv==1.1.0 \
    email_validator==2.2.0 \
    bcrypt==4.3.0 \
    cryptography==44.0.2 \
    click==8.1.8 \
    tqdm==4.67.1 \
    tenacity==9.1.2 \
    backoff==2.2.1 \
    rich==14.0.0 \
    h11==0.14.0 \
    anyio==4.9.0 \
    sniffio==1.3.1

# Step 9: Fix missing core dependencies (install WITH deps checking)
RUN pip install --no-cache-dir \
    typing_extensions \
    annotated-types \
    packaging \
    certifi \
    charset-normalizer \
    idna \
    six \
    python-dateutil \
    pytz \
    regex \
    joblib

# Optional packages
RUN pip install --no-cache-dir --no-deps \
    openai==1.82.0 \
    haystack-ai==2.14.0 || echo "Optional packages failed, continuing..."

# Create user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Environment
ENV PATH="/home/app/.local/bin:/usr/local/bin:$PATH"
ENV PYTHONPATH="/home/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV USE_GPU=False

# Copy code
COPY --chown=app:app . .
RUN mkdir -p chroma_db data benchmark/results logs tmp

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]